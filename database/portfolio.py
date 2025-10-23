#!/usr/bin/python3

import datetime
from pymongo import MongoClient
import logging
logger = logging.getLogger(__name__)



client = MongoClient('mongodb://localhost:27017/')


db = client.portfolios_database

def save_portfolio(portfolio,name,**kwargs):
    try: 
      d=dict(kwargs)
      amount=d["amount"]
      assets=portfolio.assets
      weights=portfolio.weights
      amounts=[amount*w for w in weights]
      mtms=list(portfolio.data["Adj Close"].values[-1])
      quantities=[p[0]/p[1] for p in zip(amounts,mtms)]
      #print(mtms)
      d.update({
            "name":name,
            "assets":assets,
            "weights":weights,
            "quantities":quantities,
            "created_at": datetime.datetime.utcnow()

        })
      #print(d)
      db.portfolios.insert_one(d)
    except Exception as e:
        logger.error(f"Failed to save portfolio {name}: {e}")
        raise
    
def get_portfolios(**kwargs):
   try:
     filter=dict(kwargs)
     portfs=db.portfolios.find(filter)
     return portfs
     #db.portfolios.insert_one(d)
   except Exception as e:
        logger.error(f"Failed to get portfolios: {e}")
        return []



def get_single_portfolio(name:str):

    portf=db.portfolios.find_one({"name":name})
    return portf

if __name__=='__main__':

    print(list(get_portfolios(method="risk")))
