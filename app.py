from flask import Flask,render_template,send_file,make_response
from covid19_tracker_V1 import Covid19Tracker
from flask import *
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
#import folium
import io
import random
from flask import Response
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app=Flask(__name__)

@app.route("/")
def show_charts():
    covid_obj=Covid19Tracker()
    covid_obj.scrape_world_data()
    covid_obj.scrape_india_data()
    world_summary= covid_obj.plot_world_summary()
    india_summary= covid_obj.plot_india_summary()
    world_confirmed_cases=covid_obj.plot_world_confirmed_cases()
    world_reported_deaths=covid_obj.plot_world_total_deaths_cases()
    india_agewise = covid_obj.plot_india_age_group()
    india_states_table=covid_obj.plot_india_states_table()
    india_total_cases=covid_obj.plot_india_total_cases()
    india_states_cases=covid_obj.plot_state_details()
    
    return render_template('index.html', world_summary_c1=world_summary, india_summary_c1=india_summary,
            world_confirmed_cases_c1=world_confirmed_cases, world_reported_deaths_c1=world_reported_deaths,
            india_agewise_c1=india_agewise,india_states_table_t1=india_states_table,
            india_total_cases_c1=india_total_cases, india_states_cases_c1=india_states_cases
    )

    #pipreqs <root_folder_name> generates only the required dependencies
    # return render_template('index.html',  world_reported_deaths_c1=world_reported_deaths)
    # return render_template('index.html',  india_summary_c1=india_summary)
    #basic
    #return render_template("base.html")

if __name__=="main":
    app.jinja_env.cache = {}
    app.run(debug=True)
