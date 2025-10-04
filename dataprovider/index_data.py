
import sys
import os 

#path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,path)



import pandas as pd
import dask
import dask.dataframe as dd
from dask.dataframe import from_pandas

from ml import nlp
      
def panda2_json(df:pd.DataFrame,file:str):
    json_file=f"{file}.json"
    

    df.to_json(json_file,orient="records")

def json2_pandas(file:str):
    json_file=f"{file}.json"
    from_json_file=pd.read_json(json_file,
    orient="records", 
    lines=False)
    return from_json_file

def get_major_indexes():
    return json2_pandas("dataprovider/jsons/major_indices")

def get_wiki_indexes():
    return json2_pandas("dataprovider/jsons/wiki_indexes")


def save_contries_codes():
   df= pd.read_html("https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2")[4]
   panda2_json(df,"dataprovider/jsons/countries")

def get_contry_by_code(code:str):

    df=json2_pandas("dataprovider/jsons/countries")
    resuslts=df.query(f"Code.str.lower() =='{code.lower()}'")
    return resuslts["name"].values[0]

def get_contry_code(country:str):
    df=json2_pandas("dataprovider/jsons/countries")
    mask=df["name"].apply(lambda x:nlp.compute_cosine_similarity(x.lower(),country.lower())>0.5)
    
    print(df[mask])
    resuslts=df.query(f"name.str.contains('{country.lower()}', case=False)")
    return resuslts[["name","Code"]].values

def get_region_by_code(code:str):

    df=json2_pandas("dataprovider/jsons/regions")
    resuslts=df.query(f"code.str.lower() =='{code.lower()}'")
    return resuslts["region"].values[0]
def get_index_symbol(index_name:str):
    df=json2_pandas("dataprovider/jsons/wiki_indexes")
    df =df[df["symbol"].notnull()]
    print(index_name)
    mask=df["name"].apply(lambda x:(nlp.compute_cosine_similarity(x.lower(),index_name.lower())>0.5) or (index_name.lower() in x.lower()))
    
    results=df[mask]
    #resuslts=df.query(f"name.str.contains('{index_name.lower()}', case=False)")
    return results[["name","symbol"]].values


if __name__=="__main__":
    print(get_contry_by_code("uk"))
