import time
import streamlit as st
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from database import get_portfolios,get_single_portfolio, save_portfolio
from dataprovider import yahoo
from factory import create_portfolio_by_name
from portfolio import Portfolio
from describe import summary

from viz import plot_markowitz_curve,plot

@st.cache_data(persist=True)
def cached_get_sectors_weights(assets,weights):
    sector_we=yahoo.get_sectors_weights(assets,weights)
    return sector_we

def plot_composition_pie(weights,assets):

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].pie(weights, labels=assets, autopct='%1.1f%%',
        pctdistance=0.8, startangle=90,textprops={'size': 'smaller'})
    #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    
    sector_we=cached_get_sectors_weights(assets,weights)
    axs[1].pie(sector_we.values(), labels=sector_we.keys(), autopct='%1.1f%%',
        pctdistance=0.8, startangle=90,textprops={'size': 'smaller'})
    #ax2.axis('equal')
    st.pyplot(fig)
    
    plt.close(fig)

def build_discretionnary(assets,data):
    weights=st.text_input("enter weights of tickers (seperated by coma)")
    weights=weights.split(",")
    weights=[w.strip(" ") for w in weights]
    weights=map(float,weights)
    if abs(sum(weights) - 1.0) > 1e-6:
       st.error(f"Weights must sum to 1.0, got {sum(weights):.4f}")
       return
    porfolio=Portfolio(assets,data)
    porfolio.set_weights(weights)



class Builder:
  def __init__(self) -> None:
      
      self.portfolio:Portfolio=None

  def build(self,assets,data):
    self.portfolio=Portfolio(assets,data)
    weights_str=st.text_input("enter weights of tickers (seperated by coma)")
    
    if weights_str:
        weights=weights_str.split(",")
        weights=[w.strip(" ") for w in weights]
        weights=list(map(float,weights))
        if abs(sum(weights) - 1.0) > 1e-6:
          st.error(f"Weights must sum to 1.0, got {sum(weights):.4f}")
          return
        
        self.portfolio.set_weights(weights)
        self.explore("discretionnary")

       
  def save(self,model,**kwargs):
      name=st.text_input("portfolio name")
      initial_amount=st.number_input("amount")
      if name:
        st.write(name.title())
        save_button= st.button("save")  
        if save_button:
            #st.write(self.model,name,initial_amount)
            save_portfolio(self.portfolio,name,model=model,amount=initial_amount,**kwargs)
  
  def summary(self):

      st.markdown("## summary")
      data= summary(self.portfolio,cmline=False)
      tab=pd.DataFrame()
      tab["key"]=list(data.keys())
      tab["value"]=list(data.values())
      st.table(tab)

  def composition(self):

        weights=self.portfolio.weights
        assets=self.portfolio.assets
        plot_composition_pie(weights,assets)
            
  def plot(self,model):
      plot(self.portfolio.assets,model,"1mo",show=False)
  def explore(self,model,**kwargs):

    my_funcs=["composition","summary","plot","save",]
    menu_func = option_menu(None,my_funcs,icons=None,default_index=0, orientation="horizontal")
    
    if menu_func:
        if menu_func=="composition":
        
            self.composition()
        if menu_func=="summary":
            self.summary()
                
        if menu_func=="save":
            self.save(model,**kwargs)
        if menu_func=="plot":
            self.plot("risk0")
           
      

class markowitzBuilder(Builder):
    
    def build(self,assets,data):
            
        methods=["unsafe","risk","return","sharp","compare"]
        methods_select= st.selectbox("methods",methods)
        
        if methods_select:
                #st.write(methods_select)
            if len(assets)>0:
                if methods_select=="compare":
                    plot_markowitz_curve(assets,n=3000,data=data,show=False)
                    st.pyplot(fig=plt)
                else:
                    self.build_method(methods_select,assets,data)

    def build_method(self,method_select,assets,data):
        
        if method_select=="risk":
            risk_tolerance=st.number_input("risk tolerance",key="risk")
            if risk_tolerance > 0.00:
                self.portfolio=create_portfolio_by_name(assets,method_select,data,risk_tolerance=risk_tolerance)
                
                self.explore("markowitz",method=method_select,risk_tolerance=risk_tolerance)

        elif method_select=="return":
            expected_return=st.number_input("expected return",key="return")
            if expected_return > 0.00:
                self.portfolio=create_portfolio_by_name(assets,method_select,data,expected_return=expected_return)
                
                self.explore("markowitz",method=method_select,expected_return=expected_return)
        else :
            self.portfolio=create_portfolio_by_name(assets,method_select,data)
            
            self.explore("markowitz",method=method_select)

class naiveBuilder(Builder):

     def build(self, assets, data):

         self.portfolio=Portfolio(assets, data)
         weights=[]
         n=len(assets)
         w=1/n
         for i in range(n):
             weights.append(w)
         self.portfolio.set_weights(weights)
         self.explore("naive")

class betaWeightedBuilder(Builder):

    def build(self, assets, data):
        self.portfolio=Portfolio(assets,data)
        betas=list(map(float,yahoo.get_assets_beta(assets)))
        betas_sum=sum(betas)
        weights=[beta/betas_sum for beta in betas]
        self.portfolio.set_weights(weights)
        self.explore("betaWeighted")
        
st.set_page_config(page_title="porfoli view",layout="centered")

class portfolioModel:

    def __init__(self,_name,assets,data) -> None:

        self._name=_name
        self.assets=assets
        self.data=data
        self.builder:Builder=None

    def get_assets():

        pass
    
    def get_builder(self):

        if self._name=="discretionnary":
            self.builder=Builder()
            self.builder.build(self.assets,self.data)
            #print(self.builder.porfolio)
            #time.sleep(1)
            #self.builder.explore("discretionnary")
            

        if self._name=="markowitz":

            self.builder=markowitzBuilder()
            self.builder.build(self.assets,self.data)
                 
        if self._name=="naive":

            self.builder=naiveBuilder()
            self.builder.build(self.assets,self.data)                    
        
        if self._name=="betaweighted":

            self.builder=betaWeightedBuilder()
            self.builder.build(self.assets,self.data)

    

def build():

    models=["markowitz","discretionnary","naive","betaweighted"]
    st.title("porfolio view")

    tickers=st.text_input("enter assets tickers (sepearated by coma)")
    if tickers:
        
        assets=tickers.split(",")
        assets=[a.strip(" ") for a in assets]

        data=yahoo.retrieve_data(tuple(assets))
        #print(data)
        
        model_select=st.selectbox("model",models,key="markowitz")
        
        if model_select:
            
            the_model=portfolioModel(model_select,assets,data=data)
            the_model.get_builder()


def explore():

    st.title("porfolio view")
    st.sidebar.header("portfolio selector")
    
    portfolios=get_portfolios()
    portfolios_name=[p["name"] for p in list(portfolios)]
    select = st.sidebar.selectbox('Select portfolio',portfolios_name)
    portfolio=get_single_portfolio(select)
    st.subheader(select)
    st.write("model:",portfolio["model"])
    st.write("method:",portfolio["method"])
    amount=portfolio["amount"]
    #print(portfolio)
    portfolio_assets=zip(list(portfolio["assets"]),list(portfolio["weights"]),list(portfolio["quantities"]))
    data=yahoo.retrieve_data(tuple(portfolio["assets"]),"1d")
    #print(data)
    portfolio_comp=[{"asset":p[0],"weight":p[1],"amount":amount*p[1],"quantity":p[2]} for p in portfolio_assets]
    
    portfolio_dataframe=pd.DataFrame(portfolio_comp)
    print(data.columns)
    # Handle both single and multiple rows
    if len(data) > 0:
          portfolio_dataframe["close"] = data["Adj Close"].iloc[-1].values
    else:
          st.error("No data available")
          return
   
    portfolio_dataframe["mtm"]=portfolio_dataframe["close"]*portfolio_dataframe["quantity"]
    mtm=portfolio_dataframe["mtm"].sum()
    pch=round((mtm-amount)*100/amount,2)
    delta=round(mtm-amount,2)
    st.metric("MTM",f"{round(mtm,2)}({pch}%)",delta=delta)
   
    st.markdown("### portfolio composition")
    st.dataframe(portfolio_dataframe) # will display the dataframe
    
    weights=portfolio["weights"]
    assets=portfolio["assets"]
    plot_composition_pie(weights,assets)
   
    

pages=["build","explorer"]
func = option_menu(None,pages,icons=None,default_index=0, orientation="horizontal")

if func=="explorer":
    explore()
if func=="build":
    build()
