from .MakeDatasetForM6 import *
from .DatasetFrame import *
from .DatasetFrame_2 import *
from .DatasetFrame_3 import *
from .DatasetFrame_target import *
from .model_util import *

def scaler_info_csv(asset_scaler, index_scaler, Window_size, asset_list, index_list,path=".",scaler_info = "minmax"):
    """
    how to use? : {min-max} : (X * scale_) + min_ = scaled_X
                  {standard} : (X - mean_) / scale_ = scaled_X
    """
    if scaler_info == "minmax":
        asset_scaler_info = pd.DataFrame([asset_scaler.scale_,asset_scaler.min_]).T
        asset_scaler_info.columns = ["scale_","min_"]
        
        index_scaler_info = pd.DataFrame([index_scaler.scale_,index_scaler.min_]).T
        index_scaler_info.columns = ["scale_","min_"]
    elif scaler_info == "standard":
        asset_scaler_info = pd.DataFrame([asset_scaler.mean_,asset_scaler.scale_]).T
        asset_scaler_info.columns = ["mean_","scale_"]
        
        index_scaler_info = pd.DataFrame([index_scaler.mean_,index_scaler.scale_]).T
        index_scaler_info.columns = ["mean_","scale_"]
    else:
        raise ValueError(f'{scaler_info} : not avaliable value. try one of ["minmax", "standard"]')
    
    asset_scaler_info.index = asset_list
    index_scaler_info.index = index_list
    
    os.makedirs(path+f"/scaler_info/window_{Window_size}",exist_ok = True)
    
    asset_scaler_info.to_csv(path+f"/scaler_info/window_{Window_size}/asset_scaler.csv")
    index_scaler_info.to_csv(path+f"/scaler_info/window_{Window_size}/index_scaler.csv")
    
    return asset_scaler_info,index_scaler_info
