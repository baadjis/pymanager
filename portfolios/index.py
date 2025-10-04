from dataprovider import yahoo

import yahoo_fin.stock_info as si

class Index:

      def __init__(self,ticker) -> None:
          self.ticker=ticker

      @property
      def components(self):
          """get index components

          Returns:
              list: components names 
          """
          df=yahoo.get_index_components(self.ticker)
          symbols = df['Symbol'].values.tolist()

          return symbols

      @property
      def marketcaps(self):
          
          comp=self.components

          return {x:(si.get_quote_table(x)["Market Cap"])for x in comp}
      
      def marketcap(self):

          marketcaps=self.marketcaps
          mkcs=list(marketcaps.values())
          mkc=map(float,[mk.split("B")[0] for mk in mkcs])
          
          return str(sum(list(mkc)))+"B"

          
      def marketcapw(self):
          
          summs=float(self.marketcap().split("B")[0])
          comp=self.components
          marketcaps=self.marketcaps
          weight={x:float(marketcaps[x].split("B")[0])/summs for x in comp}
          
          return weight

if __name__=='__main__':

    cac40=Index("FCHI")
    #comp=cac40.get_components()
    print(cac40.marketcapw())



        