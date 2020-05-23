import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup # for web scraping (HTML/XML parser)
import pandas as pd # Dataset handling
import numpy as np # support for calculations and pandas
import seaborn as sns # for visualization
import matplotlib.pyplot as plt # for visualization
import plotly # for visualization
import plotly.figure_factory as ff 
import plotly.express as px
import plotly.graph_objects as go

url_covid19_world='https://www.worldometers.info/coronavirus/'
url_covid19_India='https://www.mohfw.gov.in/'

class Covid19Tracker:
    
    def __init__(self):
        # pandas dataframe for processing and visualization
        self.df_all_countries = None
        self.world_summary=[]
        self.india_summary=[]
        self.df_india_summary = None
        self.df_indian_states = None
    
    def get_world_summary(self):
        return self.world_summary
    
    def get_countries_data(self):
        return self.df_all_countries
    
    def get_indian_states_data(self):
        return self.df_indian_states
    
    def world_total(self):
        return self.world_summary
    """
    ***************************************************************************************
    *** scrape_world_data(): Extracting realtime Worldwide covid-19 data
    ****************************************************************************************
    """
    def scrape_world_data(self):
        try:
            response = requests.get(url_covid19_world)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}') 
        except Exception as err:
            print(f'Other error occurred: {err}')  
        else:
            soup=BeautifulSoup(response.text, "html.parser")
            #debug : Confirming that accessing the correct webpage
            #display(soup.title)
            
            """
            Extracting Worldwide parameters: Confirmed Cases, New Cases, Reported Deaths, New Deaths
            """
            # Processing the required data from the first table that is retrieved
            coronatable=soup.find_all("table")[0]
            
            """
            Extracting each Country parameters 
            """
            c_name               = []
            c_total_cases        = []
            c_active             = []
            c_total_recovered    = []
            c_new_cases          = []
            c_new_deaths         = []
            c_total_deaths       = []
                        
            rows=coronatable.find_all("tr")[9:-8]
                        
            for row in rows:
                col=row.find_all("td")
                c_name.append(col[1].text.strip())
                c_total_cases.append(col[2].text.strip().replace(',','').replace('+',''))
                c_new_cases.append(col[3].text.strip().replace(',','').replace('+',''))
                c_total_deaths.append(col[4].text.strip().replace(',','').replace('+',''))
                c_new_deaths.append(col[5].text.strip().replace(',','').replace('+',''))
                c_total_recovered.append(col[6].text.strip().replace(',','').replace('+',''))
                c_active.append(col[7].text.strip().replace(',','').replace('+',''))


                self.df_all_countries = pd.DataFrame(list(zip(c_name, c_total_cases, c_active, c_total_recovered, c_new_cases,    
                                c_new_deaths, c_total_deaths)),columns=["Country", "Total_Cases", "Active", "Recovered", 
                                                                        "New_Cases", "New_Deaths", "Total_Deaths" ])
            """
            Analysing and Preparing the data
            """

            # df_all_countries.style.hide_index()
            # display(df_all_countries.head(10))
            # display(df_all_countries.shape)

            """
            Checking for missing/null values.
            """
            # To begin processing the dataframe. Handling the missing values by first replacing it with NaN. Python recoginizes 
            # only NaN and None as NA values. Also we can see invalid text 'N/A' for some of the parameters.

            # Checking whether parameter 'Country' is not missing values or set to invalid text 'N/A'. Else we have to drop that row.
            # print(df_all_countries.Country.isna().sum())
            # print((df_all_countries.Country == 'N/A').sum())
            # We dont have any such above rows in this dataset            
            
            # Replacing missing values with NaN
            self.df_all_countries.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # Replacing invalid text 'N/A' with NaN
            self.df_all_countries.replace('N/A', np.nan, inplace=True)

            # Checking if duplicate rows exist. Looks like there is none¶
            # print("Count Duplicate rows with country: ", df_all_countries.Country.duplicated().sum())
            # print("Count Duplicate rows: " , df_all_countries.duplicated().sum())

            """
            Missing data imputation
            """
            
            #Following two parameters vary daily. We cannot fill this value with mean/mode/median. Replacing NaN with zero. 
            self.df_all_countries.New_Cases.fillna(0, inplace = True)
            self.df_all_countries.New_Deaths.fillna(0, inplace = True)

            """
            For parameters Active cases, Total Deaths and Recovered cases:
            Imputed these values only for those countries not missing any data in the required parameters used for calculation. 
            Otherwise replacing with 0 so as to avoid generating incorrect data. Verified with the dataset. 
            Observe the count of missing values for each of this above parameter. I begin imputing by selecting the parameter which
            has the highest number of missing values.
            Relation between the parameters: Total cases = Recovered cases + Active Cases + Total Deaths
            """

            # We might have to iterate the data cleaning steps until the relevant amount of data is filled with intended values.          
            """
            Imputing Total death cases having NaN values
            """
            # Get the row indexes of countries having NaN values in the column 'Total_Deaths'
            result_df = self.df_all_countries[self.df_all_countries.Total_Deaths.isna()]
            #display (result_df)

            # --- for debugging begins --- 
            # indexes = []
            # for index1 in result_df.index:
            #     indexes.append(index1) 
            # print(indexes)    
            # # --- for debugging end --- 


            for index1 in result_df.index:
                if ((result_df['Recovered'][index1] is np.nan) or (result_df['Active'][index1] is np.nan)):
                    self.df_all_countries['Total_Deaths'][index1] = 0
                else:
                    self.df_all_countries['Total_Deaths'][index1] = int(self.df_all_countries['Total_Cases'][index1]) - \
                                                            int(self.df_all_countries['Active'][index1]) - \
                                                            int(self.df_all_countries['Recovered'][index1])
            # --- debugging  ---         
            # for i in indexes:
            #     print(self.df_all_countries['Country'][i], self.df_all_countries['Recovered'][i], 
            #           self.df_all_countries['Active'][i], self.df_all_countries['Total_Deaths'][i])
            
            """
            Imputing Recovered cases having NaN values
            """
            # Get the row indexes of countries having NaN values in the column 'Recovered'
            result_df = self.df_all_countries[self.df_all_countries.Recovered.isna()]
            #display (result_df)

            # --- for debugging begins --- 
            # indexes = []
            # for index1 in result_df.index:
            #     indexes.append(index1) 
            # print(indexes)    
            # --- for debugging end --- 

            for index1 in result_df.index:
                if ((result_df['Active'][index1] is np.nan) or (result_df['Total_Deaths'][index1] is np.nan)):
                    self.df_all_countries['Recovered'][index1] = 0
                else:
                    self.df_all_countries['Recovered'][index1] = int(self.df_all_countries['Total_Cases'][index1]) - \
                                                            int(self.df_all_countries['Active'][index1]) - \
                                                            int(self.df_all_countries['Total_Deaths'][index1])
            # --- debugging  ---         
            # for i in indexes:
            #     print(self.df_all_countries['Country'][i], self.df_all_countries['Recovered'][i], 
            #           self.df_all_countries['Active'][i], self.df_all_countries['Total_Deaths'][i])

            """
            Imputing Active cases having NaN values
            """
            # Get the row indexes of countries having NaN values in the column 'Active'
            result_df = self.df_all_countries[self.df_all_countries.Active.isna()]
            #display (result_df)

            # --- for debugging begins --- 
            # indexes = []
            # for index1 in result_df.index:
            #     indexes.append(index1) 
            # print(indexes)    
            # --- for debugging end --- 

            for index1 in result_df.index:
                if ((result_df['Recovered'][index1] is np.nan) or (result_df['Total_Deaths'][index1] is np.nan)):
                    self.df_all_countries['Active'][index1] = 0
                else:
                    self.df_all_countries['Active'][index1] = int(self.df_all_countries['Total_Cases'][index1]) - \
                                                            int(self.df_all_countries['Recovered'][index1]) - \
                                                            int(self.df_all_countries['Total_Deaths'][index1])

            # --- debugging  ---         
            # for i in indexes:
            #     print(self.df_all_countries['Country'][i], self.df_all_countries['Recovered'][i], 
            #           self.df_all_countries['Active'][i], self.df_all_countries['Total_Deaths'][i])


            #self.df_all_countries.isna().sum()
            # Convert datatype from object to int. We can perform calculations and plotting
            self.df_all_countries.New_Cases  = self.df_all_countries.New_Cases.astype(int)
            self.df_all_countries.New_Deaths = self.df_all_countries.New_Deaths.astype(int)
            self.df_all_countries.Active     = self.df_all_countries.Active.astype(int)
            self.df_all_countries.Total_Cases = self.df_all_countries.Total_Cases.astype(int)
            self.df_all_countries.Total_Deaths = self.df_all_countries.Total_Deaths.astype(int)
            self.df_all_countries.Recovered = self.df_all_countries.Recovered.astype(int)            

            #dict_world_summary = {}
            #dict_world_summary['Total_Cases']  = self.df_all_countries['Total_Cases'].sum()
            #dict_world_summary['New_Cases']    = self.df_all_countries['New_Cases'].sum()
            #dict_world_summary['Total_Deaths'] = self.df_all_countries['Total_Deaths'].sum()
            #dict_world_summary['New_Deaths']   = self.df_all_countries['New_Deaths'].sum()                        

            self.world_summary.append(self.df_all_countries['Total_Cases'].sum()) #0
            self.world_summary.append(self.df_all_countries['New_Cases'].sum()) #1
            self.world_summary.append(self.df_all_countries['Active'].sum()) #2
            self.world_summary.append(self.df_all_countries['Recovered'].sum()) #3
            self.world_summary.append(self.df_all_countries['Total_Deaths'].sum()) #4
            self.world_summary.append(self.df_all_countries['New_Deaths'].sum()) #5
    
    def plot_world_summary(self):
        labels=['Active', 'Recovered','Deceased']
        colors=['#66b3ff','lightgreen','black']
        values=[self.world_summary[2],self.world_summary[3],self.world_summary[4]]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5,textinfo='label+percent',
                                    textposition='inside', marker_colors=colors)])
      
        str1 = "Worldwide Confirmed " + format(self.world_summary[0],',d')
        fig.update_layout(showlegend=False,
                    title ={'text' : str1, 'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                    annotations=[dict(text="Worldwide",showarrow=False)]) 
        #fig.show()
        return plotly.offline.plot(fig,output_type='div')
                                      
    """
    Plotting pie visualization : Covid-19 total confirmed cases percent split in different countries 
    Plotting pie plot showing the proportion of the spread of this Pandemic across the world. Each pie is showing a few major contributor countries, and rest countries are shown as "Others". 
    Criteria: Confirmed Cases: Any countries who reported more than 100000 confirmed cases are shown and the rest of the countries are under "others" Category.
    """ 
    def plot_world_confirmed_cases(self):
        stats = [self.df_all_countries.loc[:,['Country','Total_Cases']]] #Access the entire column Country and Total_Cases
        threshold = [100000]
        for i, stat in enumerate(stats):
            df_countries = stat.groupby(["Country"]).sum()
            df_countries = df_countries.sort_values(df_countries.columns[-1],ascending= False)
            others = df_countries[df_countries[df_countries.columns[-1]] < threshold[i] ].sum()[-1]
            df_countries = df_countries[df_countries[df_countries.columns[-1]] > threshold[i]]
            df_countries = df_countries[df_countries.columns[-1]]
            df_countries["Others"] = others
            labels = [df_countries.index[i] for i in range(df_countries.shape[0])]
        
        fig=go.Figure(data=[go.Pie(labels=labels, values=df_countries, hole=.7 , textinfo='label',
                                  textposition='inside')])


        str1 = "World Confirmed Cases " + format(self.world_summary[0],',d')
        fig.update_layout(showlegend=False,
                    title ={'text' : str1, 'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                    annotations=[dict(text="Confirmed cases",showarrow=False)])   

        #fig.show()
        return plotly.offline.plot(fig,output_type='div')

    """
    Plotting pie visualization : Covid-19 total death cases percent split in different countries 
    Each pie is showing a few major contributor countries, and rest countries are shown as "Others". 
    Criteria: Confirmed Cases: Any countries who reported more than 3000 deaths are shown and the rest of the countries are under "Others" Category.
    """    
    def plot_world_total_deaths_cases(self):
        # display(self.df_all_countries.sort_values(['Total_Deaths'], ascending=False))
        stats = [self.df_all_countries.loc[:,['Country','Total_Deaths']]] #Access the entire column Country and Total_Cases
        threshold = [3000]
        for i, stat in enumerate(stats):
            df_countries = stat.groupby(["Country"]).sum()
            df_countries = df_countries.sort_values(df_countries.columns[-1],ascending= False)
            others = df_countries[df_countries[df_countries.columns[-1]] < threshold[i] ].sum()[-1]
            df_countries = df_countries[df_countries[df_countries.columns[-1]] > threshold[i]]
            df_countries = df_countries[df_countries.columns[-1]]
            df_countries["Others"] = others
            labels = [df_countries.index[i] for i in range(df_countries.shape[0])]
        
        # display(df_countries)
        fig=go.Figure(data=[go.Pie(labels=labels, values=df_countries, hole=.6 , textinfo='label+percent',
                                  textposition='inside')])
        
        
        str1 = "World Reported Deaths " + format(self.world_summary[4],',d')
        fig.update_layout(showlegend=False,
                    title ={'text' : str1, 'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                    annotations=[dict(text="Reported Deaths",showarrow=False)])   

        #fig.show()
        return plotly.offline.plot(fig,output_type='div')

    """
    ***************************************************************************************
    scrape_india_data(): Scraping Covid-19 data for India
    ***************************************************************************************
    """
    def scrape_india_data(self):
        try:
            response = requests.get(url_covid19_India)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}') 
        except Exception as err:
            print(f'Other error occurred: {err}')  
        else:
            soup=BeautifulSoup(response.text, "html.parser")
            # display(soup.title)
 
            """
            Get data for the Indian states 
            """
            state_names=[]
            state_total_cases=[]
            state_total_cured=[]
            state_total_deaths=[]
            india_states_table = soup.find("table", class_="table table-striped")
            rows=india_states_table.find_all("tr")[1:34]
            #display(rows)
            for row in rows:
                col=row.find_all("td")
                #display(col[1].text, col[2].text, col[3].text, col[4].text)
                state_names.append(col[1].text.strip().replace('#','').replace(',','').replace('+',''))
                state_total_cases.append(col[2].text.strip().replace('#','').replace(',','').replace('+',''))
                state_total_cured.append(col[3].text.strip().replace('#','').replace(',','').replace('+',''))
                state_total_deaths.append(col[4].text.strip().replace('#','').replace(',','').replace('+',''))

            self.df_indian_states = pd.DataFrame(list(zip(state_names, state_total_cases, state_total_cured,  state_total_deaths)),
                                                 columns=["States","Total_Cases", "Recovered", "Deaths"])
                                                            
            
            # Check if missing values
            # df_indian_states.isna().sum()
            
            self.df_indian_states.Total_Cases = self.df_indian_states.Total_Cases.astype(int)
            self.df_indian_states.Recovered   = self.df_indian_states.Recovered.astype(int)
            self.df_indian_states.Deaths = self.df_indian_states.Deaths.astype(int)

            # Generating number of Active cases for each Indian state
            self.df_indian_states['Active'] = self.df_indian_states.Total_Cases - self.df_indian_states.Recovered - self.df_indian_states.Deaths
                
            self.df_indian_states = self.df_indian_states[['States','Total_Cases','Active','Recovered','Deaths']]
            self.df_indian_states.sort_values('Total_Cases', ascending=False, inplace=True)
            self.df_indian_states['Mortality_Rate (per 100)'] = np.round(np.nan_to_num(100* self.df_indian_states['Deaths']/self.df_indian_states['Total_Cases']),2)
            self.df_indian_states.reset_index(drop=True,inplace=True)
            self.india_summary.append(self.df_indian_states.Total_Cases.sum())
            self.india_summary.append(self.df_indian_states.Active.sum())
            self.india_summary.append(self.df_indian_states.Recovered.sum())
            self.india_summary.append(self.df_indian_states.Deaths.sum())
            #display(self.df_indian_states)

    def plot_india_summary(self):
        labels=['Active', 'Recovered','Deceased']
        colors=['#66b3ff','lightgreen','black']
        values=[self.india_summary[1],self.india_summary[2],self.india_summary[3]]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5,textinfo='label+percent',
                                     textposition='inside',marker_colors=colors)])
      
        str1 = "India Confirmed " + format(self.india_summary[0],',d')
        fig.update_layout(showlegend=False,
                    title ={'text' : str1, 'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                    annotations=[dict(text="India",showarrow=False)])        
                          
        #fig.show()
        return plotly.offline.plot(fig,output_type='div')

            
    def plot_india_states_table(self):
        df=ff.create_table(self.df_indian_states)
        #display(self.df_indian_states.style.background_gradient(cmap='Blues'))
        return plotly.offline.plot(df,output_type='div')  
        
    def plot_india_total_cases(self):
        # Dataset sourced on 16 May 2020 from: https://api.covid19india.org/csv/

        df = pd.read_csv(r'covid_india_time_series.csv')
        #display(df.head())
        #print('df.shape : ',  df.shape)
        #display(df.dtypes)
        #display("Checking null values: ", df.isna().sum())
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Total Confirmed'],
                mode='lines+markers',
                name='Confirmed', # this sets its legend entry 
                marker= dict(color='rgb(255,104,98)', line = dict(color='rgb(255,104,98)'))
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Total Recovered'], 
                mode='lines+markers',
                name='Recovered', # this sets its legend entry
                marker= dict(color='lightgreen', line = dict(color='Blue'))
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Total Deceased'], 
                mode='lines+markers',
                name='Deceased', # this sets its legend entry
                marker= dict(color='Black', line = dict(color='Black'))
            )
        )

        #fig.update_layout(
        #    title = "Trend of COVID-19 in India"
        #)
        #display(fig.show())
        return plotly.offline.plot(fig,output_type='div')
        
    def plot_state_details(self):
        # Downloaded the Covid19 state-wise dataset for India from this source on 16 May 2020 : https://api.covid19india.org/csv/ 

        df = pd.read_csv(r'covid_india_state_wise.csv')
        #display(df.head())
        # print('df.shape : ',  df.shape)
        # display(df.dtypes)
        # display("Checking null values: ", df.isna().sum())
        df = df.sort_values('Active', ascending=False)
        df.reset_index(drop=True, inplace=True)
        #df = df.head(10)
        # fig= px.bar(df,x='State',y='Confirmed', color='Confirmed', height=600)
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=df['Active'], x=df['State'], 
                name='Active',
                marker_color='#66b3ff'
            )
        )

        fig.add_trace(
            go.Bar(
                y=df['Recovered'], x=df['State'], 
                name='Recovered',
                marker_color='lightgreen'
            )
        )

        fig.add_trace(
            go.Bar(
                y=df['Deaths'], x=df['State'], 
                name='Deceased' ,
                marker_color='Black'
            )
        )

        fig.update_layout(
            #title = "Covid-19 spread across the states in India",
            barmode='stack',
            xaxis_tickangle=-45,
            autosize=False,
            width=800,
            height=800,
        )
        #fig.show()
        return plotly.offline.plot(fig,output_type='div')

    def plot_india_age_group(self):
        #Dataset source: https://www.kaggle.com/sudalairajkumar/covid19-in-india
        df=pd.read_csv(r"AgeGroupDetails.csv")
        df1= df.rename(index = {"Oct-19": "10-19"}) 
        fig = px.pie(df1, values='Percentage', names='AgeGroup', 
                 #title="India - Age-Group wise distribution of COVID-19",
                 hole=0.5
                 )
        fig.update_traces(textposition='inside', textinfo='percent+label')#, insidetextorientation='radial')
        #fig.show()
        return plotly.offline.plot(fig,output_type='div')

