import warnings
import pandas as pd

def configure_warnings():
    """Configure global warning filters."""
    # Datetime parsing warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format*')
    
    # Regex pattern warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='This pattern is interpreted as*')
    
    # Pandas chained assignment warnings
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*through chained assignment.*')
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    
    # General pandas warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pandas.core.frame')
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas.core.frame')