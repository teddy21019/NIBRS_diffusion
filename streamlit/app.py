import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image


@st.cache_data
def read_data():
    return pd.read_parquet("./DATA/map_plot.pq")

def filter_data(df, pop):
    return df.query("`POPULATION` > @pth")

df = read_data()


st.write("# Evolution of the Adoption of NIBRS System")

with st.expander("Details", False):
    st.write("""
    ### Data Source

    [FBI Crime Data Explorer](https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads)

    I downloaded the **Master File** for NIBRS for each year. Specifically, I extracted the **Batch Header**
    from each year. The batch header either starts with "B1, 2, 3" (before 2012) or "BH" (after 2012).
    I did not use data from NACJD. They did an excellent job in integrating agency data with incident data,
    but since I only need information about agencies, it turned out to be harder to handle.

    The [perfect instruction written by Jacob Kaplan](https://nibrsbook.com/overview-of-the-data.html#the-data-as-you-get-it-from-the-fbi)
    helped me a lot on understanding the encoding.

    ### Columns

    I extracted the following columns from the batch header lines:
    - `ORI` : Code for agencies
    - `CURRENT POPULATION 1`: The population for the agency or the population of the portion of the agency which is located in the county.
    - `NUMBER OF MONTHS REPORTED`: This is self explanatory.

    Other variables as well as their description can be found in the official documentation from FBI.
    There is also a series of variables called `AGENCY ACTIVITY INDICATOR {month}` that allows us to
    get the monthly dummy of whether an agency reported to NIBRS. In the following plot I only show the total number of months reported for simplicity.

    ### Coordinates
    I also merge coordinates of ORIs from the data provided by
             [The MarshallProject](https://observablehq.com/@themarshallproject/participation-in-the-fbi-national-crime-data-collection).
        It contains coordinates of 17118 agencies (I'm not sure of the criteria of filtering), including NYPD, LAPD, etc., which is used
        to depict the benchmark map in [this article](https://www.themarshallproject.org/2022/06/14/what-did-fbi-data-say-about-crime-in-2021-it-s-too-unreliable-to-tell)
    """)
    the_mp_map = Image.open("streamlit/the marshall project benchmark.png")
    st.image(the_mp_map, caption="The figure presented in The Marshall Project to demonstrate the agency participation data compiled by the Federal Bureau of Investigation on Feb. 7, 2022")


population_threshold = st.slider(label="Log Population Filter", min_value=3.0, max_value=6.0, value=4.0, step=0.5,
                                 help="The smaller the number, the more data it renders.")

pth = np.power(10, population_threshold)

df_to_plot = filter_data(df, pth)

fig = px.scatter_mapbox(
    df_to_plot,
    lat = "latitude", lon="longitude", size="size_to_plot", color="NUMBER OF MONTHS REPORTED", hover_name="agency_name_full",
    zoom= 2.7, center={'lat':37.0902, 'lon':-95.7129},
    color_continuous_scale=[(0,"#f55b5b"),(0.7, "#87c487"), (1,"#a2d5f5")],
    animation_frame="year",
    mapbox_style="carto-positron"
)
fig.update_layout(height=600, margin={"r":0,"t":20,"l":50,"b":0})
fig.update_coloraxes(colorbar={'orientation':'h', 'thickness':20, 'y': 1})

st.plotly_chart(fig, use_container_width=True)