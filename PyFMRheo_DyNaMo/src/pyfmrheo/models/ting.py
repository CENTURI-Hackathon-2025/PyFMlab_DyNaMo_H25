from json import detect_encoding
import numpy as np
from scipy.special import gamma
from lmfit import Model, Parameters
from .geom_coeffs import get_coeff
from ..utils.signal_processing import numdiff, smooth, hyp2f1_apprx
import torch
# import torch
# from torch.special import gamma as gamma_torch



# torch compatible functions
def get_coeff_torch(indenter_type, tip_param, nu):
    if indenter_type == 'sphere':
        # Sphere: Hertzian contact
        R = tip_param
        coeff = (4 / 3) * torch.sqrt(R) / (1 - nu**2)
        exp = 1.5
    elif indenter_type == 'cone':
        alpha = tip_param
        coeff = (2 / torch.pi) * torch.tan(alpha) / (1 - nu**2)
        exp = 2.0
    elif indenter_type == 'flat':
        a = tip_param
        coeff = 2 * a / (1 - nu**2)
        exp = 1.0
    else:
        raise ValueError(f"Unsupported indenter type: {indenter_type}")
    return coeff, exp

def numdiff_torch(y):
    dy = torch.diff(y)
    return torch.cat([dy[:1], dy])  # Pad to preserve length

def smooth_torch(y, window=5):
    if window < 2:
        return y
    kernel = torch.ones(window, device=y.device) / window
    padding = window // 2
    y_padded = torch.nn.functional.pad(y, (padding, padding), mode='replicate')
    y_smoothed = torch.nn.functional.conv1d(
        y_padded.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=0
    )
    return y_smoothed.view(-1)

def hyp2f1_apprx_torch(a, b, c, z):
    # Simple approximation: 1 + ab/c z
    return 1 + (a * b / c) * z

def SolveAnalytical_torch(ttc, trc, t1, model_probe, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag):
    if model_probe == 'paraboloid':
        Cp = 1.0 / geom_coeff
        # gamma_1 = gamma_torch(1 - betaE)
        g1 = (1 - betaE)
        gamma_1 = torch.tensor(gamma(g1.cpu().numpy()))
        # gamma_2 = gamma_torch(2.5 - betaE)
        g2 = (2.5 - betaE )
        gamma_2 = torch.tensor(gamma(g1.cpu().numpy()))

        Ftp = (3 / 2) * v0t**1.5 * E0 * t0**betaE * torch.sqrt(torch.pi) * gamma_1 / (Cp * 2 * gamma_2) * ttc**(1.5 - betaE)

        if torch.abs(v0r - v0t) / v0t < 0.01:
            hyp = hyp2f1_apprx_torch(1, 0.5 - betaE, 0.5, t1 / trc)
            Frp = 3 / Cp * E0 * v0**1.5 * t0**betaE / (3 + 4 * (betaE - 2) * betaE) \
                  * t1**(-0.5) * (trc - t1)**(1 - betaE) * (-trc + (2 * betaE - 1) * t1 + trc * hyp)
        else:
            hyp = hyp2f1_apprx_torch(1, 0.5 - betaE, 0.5, t1 / trc)
            Frp = 3 / Cp * E0 * v0t**1.5 * t0**betaE / (3 + 4 * (betaE - 2) * betaE) \
                  * t1**(-0.5) * (trc - t1)**(1 - betaE) * (-trc + (2 * betaE - 1) * t1 + trc * hyp)

        return torch.cat([Ftp, Frp])

    elif model_probe in ('cone', 'pyramid'):
        Cc = 1.0 / geom_coeff
        denom = 2 - 3 * betaE + betaE**2
        if torch.abs(v0r - v0t) / v0t < 0.01:
            Ftc = 2 * v0**2 * E0 * t0**betaE / (Cc * denom) * ttc**(2 - betaE)
            Frc = -2 * v0**2 * E0 * t0**betaE / (Cc * denom) * (
                (trc - t1)**(1 - betaE) * (trc + (1 - betaE) * t1) - trc**(1 - betaE) * trc
            )
        else:
            Ftc = 2 * v0t**2 * E0 * t0**betaE / (Cc * denom) * ttc**(2 - betaE)
            Frc = -2 * v0t**2 * E0 * t0**betaE / (Cc * denom) * (
                (trc - t1)**(1 - betaE) * (trc + (1 - betaE) * t1) - trc**(1 - betaE) * trc
            )
        return torch.cat([Ftc, Frc])

    else:
        raise ValueError(f"Unsupported probe: {model_probe}")
    
def SolveNumerical_torch(delta, time_, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr):
    device = delta.device
    delta0 = delta - delta[idxCt[0]]

    delta_Uto_dot = torch.zeros_like(delta0)
    A = smooth(numdiff(delta0[idxCt]**geom_exp), smooth_w)
    A = torch.cat([A, smooth(numdiff(delta0[idxCr[0]:]**geom_exp), smooth_w)])
    delta_Uto_dot[idxCt[0]:idxCt[0]+len(A)] = A[:len(delta_Uto_dot[idxCt[0]:])]

    delta_dot = torch.zeros_like(delta0)
    B = smooth(numdiff(delta0[idxCt]), smooth_w)
    B = torch.cat([B, smooth(numdiff(delta0[idxCr[0]:]), smooth_w)])
    delta_dot[idxCt[0]:idxCt[0]+len(B)] = B[:len(delta_dot[idxCt[0]:])]

    Ftc = torch.zeros(len(idxCt), device=device)
    for i in range(len(idxCt)):
        if i < 2:
            continue  # Skip first point
        idx = torch.arange(idxCt[0]+1, idxCt[0]+i, device=device)
        if len(idx) > 0:
            Ftc[i] = geom_coeff * E0 * torch.sum(delta_Uto_dot[idx] * time_[idx].flip(0)**(-betaE))

    Frc = torch.zeros(len(idxCt), device=device)
    for j in range(idx_tm + 1, idx_tm + len(idxCt)):
        start = j - 1
        stop = idxCt[1] - 1
        if start <= stop:
            continue
        phi_range = torch.arange(start, stop - 1, -1, device=device)
        phi0 = torch.cumsum((time_[phi_range]**(-betaE)) * delta_dot[phi_range + 2], dim=0).flip(0)
        idx_min_phi0 = torch.argmin(torch.abs(phi0))
        idxCr0 = torch.arange(j+1, j-idx_min_phi0.item(), -1, device=device)
        t10 = time_[idxCr0]
        idx = torch.arange(idxCt[0]+1, idxCt[0]+1+idx_min_phi0.item(), device=device)
        Frc[j - idx_tm - 1] = geom_coeff * E0 * torch.trapz(delta_Uto_dot[idx] * t10**(-betaE), t10)

    return torch.cat([Ftc, Frc])

class TingModel:
    def __init__(self, ind_geom, tip_param, modelFt) -> None:
        # Tip geomtry params
        self.ind_geom = ind_geom         # No units
        self.tip_parameter = tip_param   # If radius units is meters, If half angle units is degrees
        self.modelFt = modelFt
        self.fit_method = 'leastsq'
        # Compiutation params
        self.fit_hline_flag = False
        self.apply_bec_flag = False
        self.bec_model = None
        # Model params #####################
        self.n_params = None
        # Scaling time
        self.t0 = 0
        # Apparent Young's Modulus
        self.E0 = 1000
        self.E0_init = 1000
        self.E0_min = 0
        self.E0_max = np.inf
        # Time of contact
        self.tc = 0
        self.tc_init = 0
        self.tc_max = 0
        self.tc_min = 0
        # Fluidity exponent
        self.betaE = 0.2
        self.betaE_init = 0.2
        self.betaE_min = 0.01
        self.betaE_max = 1
        # Contact force
        self.F0 = 0
        self.F0_init = 0
        self.F0_min = -np.inf
        self.F0_max = np.inf
        # Poisson ratio
        self.poisson_ratio = 0.5
        # Viscous drag factor
        self.vdrag = 0
        # v0t
        self.v0t = None
        # v0r
        self.v0r = None
        # Smooth window
        self.smooth_w = None
        # Moximum indentation time
        self.idx_tm = None
    

    def build_params(self):
        params = Parameters()
        params.add('E0', value=self.E0_init, min=self.E0_min, max=self.E0_max)
        params.add('tc', value=self.tc_init, min=self.tc_min, max=self.tc_max)
        params.add('betaE', value=self.betaE_init, min=self.betaE_min, max=self.betaE_max)
        params.add('F0', value=self.F0_init, min=self.F0_min, max=self.F0_max)
        return params

    def SolveAnalytical(self, ttc, trc, t1, model_probe, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag):
        # TO DO: ADD REFERENCE!!!
        # Paraboloidal geometry
        if model_probe == 'paraboloid':
            Cp=1/geom_coeff
            Ftp=3/2*v0t**(3/2)*E0*t0**betaE*np.sqrt(np.pi)*np.array(gamma(1-betaE), dtype=float)/(Cp*2*np.array(gamma(5/2-betaE), dtype=float))*ttc**(3/2-betaE)
            if np.abs(v0r-v0t)/v0t<0.01:
                Frp=3/Cp*E0*v0**(3/2)*t0**betaE/(3+4*(betaE-2)*betaE)*t1**(-1/2)*(trc-t1)**(1-betaE)*\
                    (-trc+(2*betaE-1)*t1+trc*hyp2f1_apprx(1, 1/2-betaE, 1/2, t1/trc))
            else:
                Frp=3/Cp*E0*v0t**(3/2)*t0**betaE/(3+4*(betaE-2)*betaE)*t1**(-1/2)*(trc-t1)**(1-betaE)*\
                    (-trc+(2*betaE-1)*t1+trc*hyp2f1_apprx(1, 1/2-betaE, 1/2, t1/trc))
            return np.r_[Ftp, Frp]
        # Conical/Pyramidal geometry
        elif model_probe in ('cone', 'pyramid'):
            Cc=1/geom_coeff
            if np.abs(v0r-v0t)/v0t<0.01:
                Ftc=2*v0**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
                Frc=-2*v0**2.*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                    trc**(1-betaE)*(trc))
            else:
                Ftc=2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
                Frc=-2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                    trc**(1-betaE)*(trc))
            return np.r_[Ftc, Frc]
    
    def SolveNumerical(self, delta, time_, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr):
        delta0 = delta - delta[idxCt[0]]
        delta_Uto_dot = np.zeros(len(delta0))
        A = smooth(np.r_[numdiff(delta0[idxCt]**geom_exp), numdiff(delta0[idxCr[0]:]**geom_exp)], smooth_w)
        if len(A) < len(delta_Uto_dot[idxCt[0]:]):
            A = np.append(A, A[-1])
        delta_Uto_dot[idxCt[0]:] = A
        delta_dot = np.zeros(len(delta0))
        B = smooth(np.r_[numdiff(delta0[idxCt]), numdiff(delta0[idxCr[0]:])], smooth_w)
        if len(B) < len(delta_Uto_dot[idxCt[0]:]):
            B = np.append(B, B[-1])
        delta_dot[idxCt[0]:] = B
        Ftc = np.zeros(len(idxCt))
        for i in range(len(idxCt)):
            idx = idxCt[0] + np.arange(1, i)
            Ftc[i] = geom_coeff * E0 * np.sum(delta_Uto_dot[idx]*np.flipud(time_[idx])**(-betaE))
        idx_min_phi0 = np.zeros(len(idxCt))
        Frc = np.zeros(len(idxCt))
        for j in range(idx_tm+1, idx_tm+len(idxCt)):
            phi0 = np.flipud(np.cumsum(np.flipud(time_[j-1:idxCt[1]-1:-1]**(-betaE)*delta_dot[idxCt[1]+1:j+1]), axis=0))
            phi0 = phi0[:len(idxCt)]
            idx_min_phi0 = np.argmin(np.abs(phi0))
            idxCr0 = np.arange(j+1, j-idx_min_phi0+1, -1)
            t10 = time_[idxCr0]
            idx = np.arange(idxCt[0]+1, idxCt[0]+idx_min_phi0+1)
            Frc[j-idx_tm-1] = geom_coeff * E0 * np.trapz(delta_Uto_dot[idx]*t10**(-betaE))
        return np.r_[Ftc, Frc]
    
    def model(
        self, time, E0, tc, betaE, F0, t0, F, delta, modelFt, vdrag,
        idx_tm=None, smooth_w=None, v0t=None, v0r=None
        ):
        # Get indenter shape coefficient and exponent
        geom_coeff, geom_exp = get_coeff(self.ind_geom, self.tip_parameter, self.poisson_ratio)
        # Shift time using t at contact.
        time=time-tc
        # print(f'time --> {time}')
        # print(f'tc --> {tc}')
        # Compute deltat.
        deltat=time[1]-time[0]
        # print(f'idx_tm --> {idx_tm}')
        # If no t max index is given search the index of F max.
        if idx_tm is None:
            idx_tm = np.argmax(F)
        # Get t max value.
        tm = time[idx_tm]
        # print(f'tm --> {tm}')
        # Determine non contact trace region.
        idxNCt=np.where(time<0)[0]
        # Determine contact trace region
        idxCt=np.where(time>=0)[0]
        # Get indices corresponding to contact trace region.
        # Including t max.
        idxCt = np.arange(idxCt[0], idx_tm + 1)
        # Determine contact time trace.
        ttc=time[idxCt]
        if v0t is None and self.v0t is None:
            # Define range to compute trace speed.
            # Including t max.
            range_v0t=np.arange((idx_tm-int(len(ttc)*3/4)), idx_tm)
            # Fit 1 degree polynomial (x0 + m) to trace and retrace for determining
            # the corresponding speeds (x0)
            v0t = np.polyfit(time[range_v0t], delta[range_v0t], 1)[0]
            self.v0t = v0t
        elif v0t is None and self.v0t is not None:
            v0t = self.v0t
        if v0r is None and self.v0r is None:
            # Define range to compute retrace speed.
            # Excluding t max.
            range_v0r=np.arange(idx_tm+2, (idx_tm+1+int(len(ttc)*3/4)))
            # Fit 1 degree polynomial (x0 + m) to trace and retrace for determining
            # the corresponding speeds (x0) 
            v0r = -1 * np.polyfit(time[range_v0r], delta[range_v0r], 1)[0]
            self.v0r = v0r
        elif v0r is None and self.v0r is not None:
            v0r = self.v0r
        # Compute mean speed.
        v0=(v0r+v0t)/2
        # Compute retrace contact time.
        # TO DO: ADD REFERENCE TO ARTICLE!!!!
        tcr=(1+v0r/v0t)**(1/(1-betaE))/((1+v0r/v0t)**(1/(1-betaE))-1)*tm
        # If the retrace contact time is smaller than t max,
        # define the end of the contact retrace region as 3 times t max.
        if not tcr<tm:
            idxCr=np.where((time>tm) & (time<=tcr))[0]
        else:
            idxCr=np.where((time>tm) & (time<=3*tm))[0]
        # print(idxCr)
        # Define in contact retrace region.
        trc=time[idxCr]
        # Compute t1
        # TO DO: ADD REFERENCE TO ARTICLE!!!!
        t1=trc-(1+v0r/v0t)**(1/(1-betaE))*(trc-tm)
        # Select only the values larger than 0 of t1.
        t1=t1[t1>0]
        # Select the region of retrace time where t1 is larger than 0.
        trc=trc[t1>0]
        # Select the retrace contact indices corresponding to the retrace
        # time region where t1 is larger than 0. 
        idxCr=idxCr[:len(trc)]
        # print(idxCr)
        # Assign the value of F0 to the non contact region.
        FtNC=F0*np.ones(idxNCt.size)
        # Compute Force according to the selected mode:
        if modelFt == 'analytical':
            FJ = self.SolveAnalytical(
                ttc, trc, t1, self.ind_geom, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag
            )
        elif modelFt == 'numerical':
            FJ = self.SolveNumerical(
                delta, time, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr
            )
        else:
            print(f'The modelFt {modelFt} is not supported. Current valid modelFt: analytical, numerical.')
        # Determine non contact retrace region.
        idxNCr=np.arange((len(FJ)+len(FtNC)+1),len(delta)+1)
        # Assign the value of F0 to the non contact region.
        FrNC=F0*np.ones(idxNCr.size)
        # Concatenate non contact regions to the contact region. And return.
        return np.r_[FtNC, FJ+F0, FrNC]+smooth(numdiff(delta)*vdrag/numdiff(time), 21)
    
    def fit(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        #self.fit_time = time;self.fit_force = F;self.fit_ind = delta
        
        # Define fixed params
        self.t0 = t0
        self.idx_tm = idx_tm
        self.smooth_w = smooth_w
        self.v0t = v0t
        self.v0r = v0r
        
        # # Define fixed params
        # fixed_params = {
        #     't0': self.t0, 'F': F, 'delta': delta,
        #     'modelFt': self.modelFt, 'vdrag': self.vdrag, 'smooth_w': self.smooth_w,
        #     'idx_tm': self.idx_tm, 'v0t': self.v0t, 'v0r': self.v0r
        # }
        
        # Use PyTorch for fitting

        # Convert time and target to tensors
        # Automatic device selection
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For M1/M2/M3 Macs
        else:
            device = torch.device("cpu")

        print("Using device:", device)

        # Convert fixed parameters to tensors on device
        fixed_params = {
            't0': torch.tensor(self.t0, dtype=torch.float32, device=device),
            'F': torch.tensor(F, dtype=torch.float32, device=device),
            'delta': torch.tensor(delta, dtype=torch.float32, device=device),
            'modelFt': self.modelFt,  # Keep as-is; it's a function
            'vdrag': torch.tensor(self.vdrag, dtype=torch.float32, device=device),
            'smooth_w': torch.tensor(self.smooth_w, dtype=torch.float32, device=device),
            'idx_tm': self.idx_tm,  # Leave as-is unless used for tensor indexing
            'v0t': torch.tensor(self.v0t, dtype=torch.float32, device=device),
            'v0r': torch.tensor(self.v0r, dtype=torch.float32, device=device)
        }
        
        time_torch_0 = torch.tensor(time, dtype=torch.float32, device=device)
        F_torch_0 = torch.tensor(F, dtype=torch.float32, device=device, requires_grad=True)

        # time_torch = time_torch_0.detach().cpu().numpy()
        # F_torch = F_torch_0.detach().cpu().numpy()

        # Get initial parameter values
        init_params = self.build_params()
        # Extract initial values for each parameter
        E0_init = init_params['E0'].value
        tc_init = init_params['tc'].value
        betaE_init = init_params['betaE'].value
        F0_init = init_params['F0'].value

        # Initialize free parameters as torch tensors with gradients
        E0 = torch.tensor([E0_init], dtype=torch.float32, requires_grad=True, device=device)
        tc = torch.tensor([tc_init], dtype=torch.float32, requires_grad=True, device=device)
        betaE = torch.tensor([betaE_init], dtype=torch.float32, requires_grad=True, device=device)
        F0 = torch.tensor([F0_init], dtype=torch.float32, requires_grad=True, device=device)

        params = [E0, tc, betaE, F0]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        print (f'lr:    {optimizer.param_groups[0]["lr"]}')
        n_iter = 50

        # Optimization loop
        for i in range(n_iter):
            optimizer.zero_grad()
            # Model prediction using self.model (must support torch tensors)
            # Convert torch tensors to numpy for model function
            F_model_np = self.model(
                time_torch_0.cpu(),
                E0.cpu().item(),
                tc.cpu().item(),
                betaE.cpu().item(),
                F0.cpu().item(),
                fixed_params['t0'].cpu().item(),
                fixed_params['F'].cpu(),
                fixed_params['delta'].cpu(),
                fixed_params['modelFt'],
                fixed_params['vdrag'].cpu().item(),
                fixed_params['idx_tm'],
                fixed_params['smooth_w'].cpu().item(),
                fixed_params['v0t'].cpu().item(),
                fixed_params['v0r'].cpu().item()
            )
            # F_model_np = self.model(
            #     time_torch_0.to(device, dtype=torch.float32),
            #     E0.to(device, dtype=torch.float32),
            #     tc.to(device, dtype=torch.float32),
            #     betaE.to(device, dtype=torch.float32),
            #     F0.to(device, dtype=torch.float32),
            #     fixed_params['t0'].to(device, dtype=torch.float32),
            #     fixed_params['F'].to(device, dtype=torch.float32),
            #     fixed_params['delta'].to(device, dtype=torch.float32),
            #     torch.from_numpy(fixed_params['modelFt']).to(device, dtype=torch.float32),
            #     fixed_params['vdrag'].to(device, dtype=torch.float32),
            #     fixed_params['idx_tm'].to(device),  # dtype depends on index type
            #     fixed_params['smooth_w'].to(device, dtype=torch.float32),
            #     fixed_params['v0t'].to(device, dtype=torch.float32),
            #     fixed_params['v0r'].to(device, dtype=torch.float32),
            # )
            # Convert model output back to torch tensor for loss calculation
            F_model = torch.tensor(F_model_np, dtype=torch.float32, device=device, requires_grad=True)
            loss = torch.mean((F_model - F_torch_0) ** 2)
            loss.backward()
            optimizer.step()

        self.n_params = 4

        # Collect fitted values into dictionary-like result
        result_ting = {
            "E0":    E0.detach().cpu().item(),
            "tc":    tc.detach().cpu().item(),
            "betaE": betaE.detach().cpu().item(),
            "F0":    F0.detach().cpu().item(),
            "loss":  loss.item(),
        }


        print (result_ting)
        print (type(result_ting))

        # Assign fit results to model params
        self.E0 = result_ting['E0']
        self.tc = result_ting['tc']
        self.betaE = result_ting['betaE']
        self.F0 = result_ting['F0']

        # Compute metrics
        modelPredictions = self.eval(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)

        absError = modelPredictions - F

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(F))

        # Get goodness of fit params
        self.chisq = self.get_chisq(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)
        self.redchi = self.get_red_chisq(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)

    def eval(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        return self.model(
            time, self.E0, self.tc, self.betaE, self.F0, t0, F, delta,
            self.modelFt, self.vdrag, idx_tm, smooth_w, v0t, v0r)

    def get_residuals(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        return F - self.eval(time, F, delta, t0,idx_tm, smooth_w, v0t, v0r)

    def get_chisq(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        a = (self.get_residuals(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)**2/F)
        return np.sum(a[np.isfinite(a)])
    
    def get_red_chisq(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        return self.get_chisq(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r) / self.n_params
    
    def fit_report(self):
        print(f"""
        # Fit parameters
        Indenter shape: {self.ind_geom}\n
        Tip paraneter: {self.tip_parameter}\n
        Model Format: {self.modelFt}\n
        Viscous Drag: {self.vdrag}\n
        Smooth Window: {self.smooth_w}\n
        t0: {self.t0}\n
        Maximum Indentation Time: {self.idx_tm}\n
        Number of free parameters: {self.n_params}\n
        E0: {self.E0}\n
        tc: {self.tc}\n
        betaE: {self.betaE}\n
        F0: {self.F0}\n
        # Fit metrics
        MAE: {self.MAE}\n
        MSE: {self.MSE}\n
        RMSE: {self.RMSE}\n
        Rsq: {self.Rsquared}\n
        Chisq: {self.chisq}\n
        RedChisq: {self.redchi}\n
        """
        )
