import numpy as np

def generate_log_data(generator, scale, n, shuffle=False, gen_type="All", **kwargs):
    exclude_area = False
    include_area = False
    drop_eta_1 = False
    drop_eta_2 = False
    add_noise = False

    match type:
        case "exclude_area":
            exclude_area = True
        case "include_area":
            include_area = True
        case "drop_eta_1":
            drop_eta_1 = True
        case "drop_eta_2":
            drop_eta_2 = True
        case "add_noise":
            add_noise = True
        case _:
            pass
    
    log_scale_1 = np.random.uniform(*scale, n)
    log_scale_2 = np.random.uniform(*scale, n)
    etas, G_s = generator((10**log_scale_1)**2, (10**log_scale_2)**2)

    if exclude_area:
        if "eta_1_range" in kwargs and "eta_2_range" in kwargs:
            eta_1_range = kwargs["eta_1_range"]
            eta_2_range = kwargs["eta_2_range"]

            eta_1_points = (etas[:, 0] < eta_1_range[0]) | (etas[:, 0] > eta_1_range[1])
            eta_2_points = (etas[:, 1] < eta_2_range[0]) | (etas[:, 1] > eta_2_range[1])

            kept_points = eta_1_points & eta_2_points

            etas = etas[kept_points]
            G_s = G_s[kept_points]

        else:
            raise ValueError("eta_1_range and eta_2_range must be specified when type is exclude_area")
    
    if include_area:
        if "eta_1_range" in kwargs and "eta_2_range" in kwargs:
            eta_1_range = kwargs["eta_1_range"]
            eta_2_range = kwargs["eta_2_range"]

            eta_1_points = (etas[:, 0] > eta_1_range[0]) & (etas[:, 0] < eta_1_range[1])
            eta_2_points = (etas[:, 1] > eta_2_range[0]) & (etas[:, 1] < eta_2_range[1])

            kept_points = eta_1_points | eta_2_points

            etas = etas[kept_points]
            G_s = G_s[kept_points]

        else:
            raise ValueError("eta_1_range and eta_2_range must be specified when type is include_area")

    if add_noise:
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
    
    
    



