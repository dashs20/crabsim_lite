from gnc.ctl import AB_live
from config.config_util import load_gnc_config

gnc_config = load_gnc_config('./config/gnc/gnc.yaml')
AB_instance = AB_live(gnc_config)