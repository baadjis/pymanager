import sys

from shell import Cli
from dataprovider import yahoo,index_data
from ml import nlp

ticker_arg = {"ticker": {"type": str, "help": "the ticker"}}
country_arg={"--country": {"type": str, "default": "","help": "The country of the indexes"}}
region_arg={"--region": {"type": str, "default": "","help": "The region of the indexes"}}
geo_arg={"--geo": {"type": str, "default": "","help": "The theme of the indexes"}}
industry_arg={"--industry": {"type": str, "default": "","help": "The industry of the indexes"}}



index_cli = Cli("index")

def index_menu(st:str):
        
    fun=None
    while True:
        
        fun=input("[index]"+st+'$')
        
        match fun:
             
          case "components":
             
            comps=yahoo.get_index_components(st)
            print(comps)

          case "quit":
              sys.exit()
          
          case _:
              
              print("unknown command")

          

@index_cli.command()
def quit():
    sys.exit()

@index_cli.command(cmd_args=ticker_arg)
def go(ticker:str):
    index_menu(ticker)

@index_cli.command()
def majors():
    print(index_data.get_major_indexes())


@index_cli.command(cmd_args={**country_arg,**region_arg,**geo_arg,**industry_arg})
def ls(country:str="",region:str="",geo:str="",industry:str=""):
    indexes=index_data.get_wiki_indexes()
    if country:
      
      cname=index_data.get_contry_by_code(country)
      
      indexes=indexes[indexes["country"].isin([value for value in indexes["country"] if str(value).lower() in cname.lower() ])]
    
    if region:
      rname=index_data.get_region_by_code(region)
      print(rname)
      indexes=indexes[indexes["region"].isin([value for value in indexes["region"] if str(value).lower() in rname.lower() ])]


    if geo:
       
       indexes=indexes.query(f"geo.str.lower()=='{geo.lower()}'")

    if industry:
       indexes=indexes[indexes["industry"].notnull()]
       indexes=indexes.query(f"industry.str.contains('{industry.lower()}',case=False)")
       
    print(indexes)
@index_cli.command()
def wiki():
   while True:
      fun=input("index[question]"+'$')
      if fun=="quit":
         sys.exit()
      

      if "code" in fun.lower():

          named_entities=nlp.get_named_entities(fun) 
          if len(named_entities)==0:
            s=fun.split(" ")
            before_last=s[:-1]
            last=s[-1].capitalize()
            before_last_joined=" ".join(before_last)
            fun_t=before_last_joined+ " "+ last
            print(fun_t)
           
            named_entities =nlp.get_named_entities(fun_t)
            
          if named_entities:
            for el in named_entities:
            
              if el[1]=="GPE":
                
                country=index_data.get_contry_code(el[0])
                print(f"{country}")

      if ("ticker" in fun.lower() or ("symbol" in fun.lower())):
                  stop_words=["what","is","ticker","symbol","of","code","the"]
                  ft=fun.lower()
                  for el in stop_words:
                     ft =ft.replace(el,'')
                  ft=ft.lstrip()
                  
                  #ft=fun.lower().replace("what","").replace("ticker","").replace("is","").replace("of","")\
                  # .replace("symbol","").replace("code","").strip()
                  symbol=index_data.get_index_symbol(ft)
                  print(f"{symbol}")

                   
      else:
          print("not found")

      


if __name__=="__main__":
    index_cli.run()