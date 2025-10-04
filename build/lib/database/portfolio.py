#!/usr/bin/python3

import datetime
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')

db = client.portfolios_database

def save_portfolio(portfolio,name,**kwargs):
     
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
    
def get_portfolios(**kwargs):

    filter=dict(kwargs)
    portfs=db.portfolios.find(filter)
    return portfs
        #db.portfolios.insert_one(d)


def get_single_portfolio(name:str):

    portf=db.portfolios.find_one({"name":name})
    return portf

if __name__=='__main__':

    print(list(get_portfolios(method="risk")))