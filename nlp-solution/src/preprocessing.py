import pandas as pd
import numpy as np
from itertools import chain




def formatting(df: pd.DataFrame) -> pd.DataFrame:
    assert "aspects" and "comment" in df, "Cannot find `aspects` and `comment` columns"
    
    df['aspects'] = df['aspects'].apply(eval)
    reformat_train = map(lambda x,y: [(x,i) for i in y],df['comment'],df['aspects'])
    
    temp = list(chain(*reformat_train))
    output_df = pd.DataFrame(temp,columns=["comment","label"])
    output_df['category'] = output_df['label'].apply(lambda x:x.split('.')[1])
    
    return output_df
  