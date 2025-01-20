import numpy as np

from sklearn.preprocessing import StandardScaler

def generate_log_data(generator, scale, n, shuffle=False, gen_type="All", noise_type=None, **kwargs):
    exclude_area = False
    include_area = False
    drop_eta_1 = False
    drop_eta_2 = False
    add_out_noise = False
    add_in_noise = False
    discriminant_condition = False

    match gen_type:
        case "exclude_area":
            exclude_area = True
        case "include_area":
            include_area = True
        case "drop_eta_1":
            drop_eta_1 = True
        case "drop_eta_2":
            drop_eta_2 = True
        case "d_condition":
            discriminant_condition = True
        case _:
            pass
    
    match noise_type:
        case "out_noise":
            add_out_noise = True
        case "in_noise":
            add_in_noise = True
        case _:
            pass
    
    log_scale_1 = np.random.uniform(*scale, n)
    log_scale_2 = np.random.uniform(*scale, n)

    if exclude_area:
        if "eta_1_range" in kwargs and "eta_2_range" in kwargs:
            eta_1_range = kwargs["eta_1_range"]
            eta_2_range = kwargs["eta_2_range"]

            eta_1_points = (log_scale_1 < eta_1_range[0]) | (log_scale_1 > eta_1_range[1])
            eta_2_points = (log_scale_2 < eta_2_range[0]) | (log_scale_2 > eta_2_range[1])

            kept_points = eta_1_points | eta_2_points

            log_scale_1 = log_scale_1[kept_points]
            log_scale_2 = log_scale_2[kept_points]

        else:
            raise ValueError("eta_1_range and eta_2_range must be specified when type is exclude_area")
    
    if include_area:
        if "eta_1_range" in kwargs and "eta_2_range" in kwargs:
            eta_1_range = kwargs["eta_1_range"]
            eta_2_range = kwargs["eta_2_range"]

            eta_1_points = (log_scale_1 > eta_1_range[0]) & (log_scale_1 < eta_1_range[1])
            eta_2_points = (log_scale_2 > eta_2_range[0]) & (log_scale_2 < eta_2_range[1])

            kept_points = eta_1_points | eta_2_points

            log_scale_1 = log_scale_1[kept_points]
            log_scale_2 = log_scale_2[kept_points]

        else:
            raise ValueError("eta_1_range and eta_2_range must be specified when type is include_area")
    
    if discriminant_condition:
        if "d_condition" in kwargs:
            cond = kwargs["d_condition"]
            discriminant = generator.get_discriminant((10**log_scale_1)**2, (10**log_scale_2)**2)

            if cond == ">=":
                kept_points = discriminant >= 0
            elif cond == ">":
                kept_points = discriminant > 0
            elif cond == "<=":
                kept_points = discriminant <= 0
            elif cond == "<":
                kept_points = discriminant < 0
            else:
                raise ValueError("d_condition must be one of >=, >, <=, <")

            log_scale_1 = log_scale_1[kept_points]
            log_scale_2 = log_scale_2[kept_points]
        else:
            raise ValueError("d_condition must be specified when type is d_condition")
    
    if add_in_noise:
        if "num_samples" in kwargs and "noise" in kwargs:
            num_samples = kwargs["num_samples"]
            noise = kwargs["noise"]
            linear_scale_1 = (10**log_scale_1)**2
            linear_scale_2 = (10**log_scale_2)**2

            G_samples = np.zeros((linear_scale_1.shape[0], 3, num_samples + 1))

            for i in range(num_samples):
                linear_scale_1_noise = linear_scale_1 + np.random.normal(0, noise, linear_scale_1.shape)
                linear_scale_2_noise = linear_scale_2 + np.random.normal(0, noise, linear_scale_2.shape)

                etas, G_s = generator(linear_scale_1_noise, linear_scale_2_noise)
                G_samples[:, :, i] = G_s
            
            etas, G_s = generator(linear_scale_1, linear_scale_2)
            G_samples[:, :, num_samples] = G_s
            
            return etas, G_samples


    etas, G_s = generator((10**log_scale_1)**2, (10**log_scale_2)**2)

    if add_out_noise:
        if "noise" in kwargs:
            noise = kwargs["noise"]
            G_s += np.random.normal(0, noise, G_s.shape)
        else:
            raise ValueError("noise must be specified when type is add_noise")
        

    if drop_eta_1:
        etas[:, 0] = 0
    if drop_eta_2:
        etas[:, 1] = 0
    


    if shuffle:
        total = np.concatenate((etas, G_s), axis=1)
        np.random.shuffle(total)
        etas = total[:, :2]
        G_s = total[:, 2:]
    
    return etas, G_s
    
    
class CustomScalerX:
    epsilon = 1e-8

    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X):
        X_new = np.log(X)

        self.scaler = self.scaler.fit(X_new)

        return self
    
    def transform(self, X):
        X_new = np.log(X)

        X_new = self.scaler.transform(X_new)

        return X_new

    def inverse_transform(self, X, y):
        x = self.scaler.inverse_transform(X)
        
        x = np.exp(X)

        return x

class CustomScalerY:
    def __init__(self):
        self.epsilon = 1e-8
        self.scaler = StandardScaler()
    
    def fit(self, Y):
        y_new = np.zeros_like(Y)
        y_new[:, 0] = np.log(-(Y[:, 0] - self.epsilon))
        y_new[:, 1] = np.log(-(Y[:, 1] - self.epsilon))
        y_new[:, 2] = np.log(Y[:, 2] + self.epsilon)

        self.scaler = self.scaler.fit(y_new)

        return self

    def transform(self, Y):
        y_new = np.zeros_like(Y)
        y_new[:, 0] = np.log(-(Y[:, 0] - self.epsilon))
        y_new[:, 1] = np.log(-(Y[:, 1] - self.epsilon))
        y_new[:, 2] = np.log(Y[:, 2] + self.epsilon)

        y_new = self.scaler.transform(y_new)

        return y_new
    
    def inverse_transform(self, Y):
        y = self.scaler.inverse_transform(Y)

        y[:, 0] = -np.exp(y[:, 0])
        y[:, 1] = -np.exp(y[:, 1])
        y[:, 2] = np.exp(y[:, 2])

        return y
    
    def inverse_transform_std(self, Y):
        y = Y * self.scaler.scale_

        y[:, 0] = -np.exp(y[:, 0])
        y[:, 1] = -np.exp(y[:, 1])
        y[:, 2] = np.exp(y[:, 2])

        return y

