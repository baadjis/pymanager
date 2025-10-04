
import requests
import pandas as pd

msciurl="https://www.msci.com/constituents"
msciwordurl="https://www.msci.com/c/portal/layout?p_l_id=1317535&p_p_cacheability=cacheLevelPage&p_p_id=indexconstituents_WAR_indexconstituents_INSTANCE_nXWh5mC97ig8&p_p_lifecycle=2&p_p_resource_id=990100"
msci_index_list_url="https://www.msci.com/c/portal/layout?p_l_id=1317535&p_p_cacheability=cacheLevelPage&p_p_id=indexconstituents_WAR_indexconstituents_INSTANCE_nXWh5mC97ig8&p_p_lifecycle=2&p_p_resource_id="

def get_msci_indexes_list():
    payload={}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # This is chrome, you can set whatever browser you like
    r=requests.get(msci_index_list_url,headers=headers)
    print(r.status_code)
    info = r.json()
    print(info)

    df = pd.json_normalize(info["indices"]) 
    return(df[["index_name","index_code"]])

def get_msci_single_index_code(ind:str):
    df=get_msci_indexes_list()
    indice=df[df["index_name"]==ind]

    if len(indice):
        return str(indice.index_code.values[0])
    else:
       print(f"{ind} not found")
       a=df[df['index_name'].str.contains(ind)]
       print("try:")
       print(a.index_name.values)
       return

def get_msci_index_components(indice:str):
    
    index_code=get_msci_single_index_code(indice)
    #print(index_code)
    if index_code:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # This is chrome, you can set whatever browser you like

        r=requests.get(msci_index_list_url+index_code,headers=headers)

        info = r.json()

        df = pd.json_normalize(info["constituents"]) 
        return(df)

if __name__=='__main__':
    get_msci_indexes_list()