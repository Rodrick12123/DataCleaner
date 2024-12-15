import pandas as pd



class CSVSampleAnalyzer:
    
    def samples_with_large_na(data,na_threshold = .2):
        rows_with_large_na = data[data.isna().sum(axis=1) >= na_threshold]
        return rows_with_large_na
    
    def samples_with_duplicate_rows(data):
        duplicate_rows = self.data[self.data.duplicated()]
        return duplicate_rows
    

        