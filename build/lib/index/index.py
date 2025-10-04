
from dataprovider import yahoo
from functools import cached_property, lru_cache
import pandas as pd
import dask
import dask.dataframe as dd
from dask.dataframe import from_pandas

import yahoo_fin.stock_info as si
from stock import Stock

def get_quote_table(x):
    return si.get_quote_table(x)

@dask.delayed
def get_market_cap(x):
    return get_quote_table(x)["Market Cap"]


def format_to_float(x):
    letter_part= list(x)[-1]
    float_part=x.split(letter_part)[0]
    multiple=1000000
    if letter_part=="B":
        multiple=1000*multiple
    if letter_part=="T":
        multiple =1000000*multiple

    return float(float_part)*multiple
@dask.delayed
def delayed_format_to_float(x):
    return format_to_float(x)
def format_to_letter(x):
    letter_part="M"
    
    mult=1000000
    float_part=x
    if x >mult :

        if x<1000*mult:
            float_part=x/mult
        else:
            mult=1000*mult 
            if x<1000*mult:
                float_part=x/mult
                letter_part="B"
            else:
                mult=1000*mult
                float_part=x/mult
                letter_part="T"

    return str(float_part)+letter_part

def market_cap_category(x):
    letter_part= list(x)[-1]
    float_part=x.split(letter_part)[0]
    if letter_part=="T":
        return "Mega"
    cap_category="Small"
    if letter_part=="B":
        
        cap=float(float_part)
        if cap>=10:
            cap_category="Large"
        else:
            if cap >=2:
                cap_category="Mid"
            
                
    return cap_category





    
class Index:

      def __init__(self,ticker) -> None:
          self.ticker=ticker

      @cached_property
      def components(self):
          """get index components

          Returns:
              list: components names 
          """
         

          return (yahoo.get_index_components(self.ticker))

      @cached_property
      def symbols(self):
        df=self.components
        return (df['Symbol'].values.tolist())

        
      @property
      def sectors(self):
        
        comps=self.symbols
        
        return [ Stock(c).get_sector() for c in comps]
      
      @property
      def industries(self):
        comps=self.symbols
        return [ Stock(c).get_industry() for c in comps]

      @property
      def marketcaps(self):
          
          comp=self.symbols
          mcaps={}
          for x in comp:
              print(x)
              mcaps[x]=get_market_cap(x)
          return dask.compute(mcaps)[0]
      
      def marketcap(self):

          marketcaps=self.marketcaps
          mkcs=list(marketcaps.values())
          mkcs_list=[]
          for mk in mkcs:
             formated_mk=delayed_format_to_float(mk)
             mkcs_list.append(formated_mk)
          mkc=dask.delayed(sum)(mkcs_list)
          
          return format_to_letter(mkc.compute())
      
    
          
      def marketcapw(self):
          
          summs=float(self.marketcap().split("B")[0])
          comp=self.symbols
          marketcaps=self.marketcaps
          weight={x:float(marketcaps[x].split("B")[0])/summs for x in comp}
          
          return weight

      def get_all_data(self):
            comps=self.components
            comps["sectors"]=self.sectors
            comps["industries"]=self.industries
            comps["marketcap"]=list(self.marketcaps.values())
            return comps
      
      def save_to_json(self):
          json_file=f"index/{self.ticker}.json"
          df= self.get_all_data()
          final_data=df.drop(columns=['Last Price', 'Change',"% Change","Volume","marketcap","Company Name"])
          final_data.to_json(json_file,orient="records")

      def lood_data(self):
        json_file=f"index/{self.ticker}.json"
        print(json_file)
        from_json_file=dd.read_json(json_file,
    blocksize=None, orient="records", 
    lines=False).compute()
        df=self.components
        #mcs=list(self.marketcaps.values())
        #print(mcs)
        #df["marketcap"]=mcs
        print(df)
        from_pandas_df=from_pandas(df, npartitions=3)
        
        joined_data=dd.merge(from_pandas_df, from_json_file, on='Symbol', how='outer')
     
        return joined_data.compute()

if __name__=='__main__':

    cac40=Index("FCHI")
    #comp=cac40.get_components()
    #cac40.save_to_json()
    print(cac40.lood_data())



        
