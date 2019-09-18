import pandas as pd
import matplotlib.pyplot as plt




class DataShower(object):
    def __init__(self,path):
        self.dataframe=pd.read_csv(path)
    
    def queryMutiple(self,dic):
        result=self.dataframe
        for key,value in dic.items():
            result=self.queryByCol(result,key,value)
        return result

    def queryByCol(self,df,col,name):
        # exp: example_ 'A>5.0 & (B>3.5 | C<1.0)'
        exp='{}=={}'.format(col,name)
        print(exp)
        return df.query(exp)

    def plot(self,dataframe,x_col,top_range):
        clean_acc=dataframe['clean_acc'].values.tolist()[:top_range]

        trojan_acc=dataframe['trojan_acc'].values.tolist()[:top_range]

        x=dataframe[x_col].values.tolist()[:top_range]

        plt.plot(x,clean_acc,label='clean_acc',linewidth=3,color='red',marker='o', markerfacecolor='red',markersize=12)
        plt.plot(x,trojan_acc,label='trojan_acc',linewidth=3,color='green',marker='v', markerfacecolor='green',markersize=12)
        
        plt.xlabel('trojan ratio')
        plt.ylabel('Accuracy(clean/trojan)')
        plt.grid(True)
        plt.legend()
        plt.savefig('log/figure_trojan_ratio.jpg')

        plt.show()





