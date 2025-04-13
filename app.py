import streamlit as st
import os
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import csv
from datetime import datetime

st.set_page_config(
    page_title="Customer Segmentation and Insights",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

user_data_file = 'users.csv'

url1 = "https://raw.githubusercontent.com/raj0929/Customer-Segmentation-and-Predictive-Insights/refs/heads/main/sales.csv"
data1 = pd.read_csv(url1)
url2 = "https://raw.githubusercontent.com/raj0929/Customer-Segmentation-and-Predictive-Insights/refs/heads/main/Mall_Customers.csv"
data2 = pd.read_csv(url2)
url3 = "https://raw.githubusercontent.com/raj0929/Customer-Segmentation-and-Predictive-Insights/refs/heads/main/Customer%20Data.csv"
data3 = pd.read_csv(url3)


if not os.path.exists(user_data_file):
    df = pd.DataFrame(columns=['Email','Password'])
    df.to_csv(user_data_file,index=False)
else:
    df = pd.read_csv(user_data_file,dtype={'Email':str,'Password':str})

df['Email'] =df['Email'].astype(str).str.strip()
df['Password'] =df['Password'].astype(str).str.strip()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

query_params = st.query_params
if 'auth' in query_params and query_params['auth'] == 'true':
    st.session_state.authenticated = True
    st.session_state.user_id = query_params.get('user',"")

with st.sidebar:
    if st.session_state.authenticated:
        st.title(f"👋 Welcome, {st.session_state.user_id}!")
        selected = option_menu(
            menu_title="Home",
            options=['About','Customer Wise Analysis', 'Region Wise Analysis', 'Product Wise Analysis','Mall Customers Analysis','Mall Customers Prediction','Market Customers Analysis','Market Customers Prediction','Contact','Logout'],
            icons=['pin','speedometer2', '', 'gear','graph-down','magic','graph-up','magic','house', 'unlock'],
            menu_icon="house",
            default_index=0,
        )
    else:
        selected =option_menu("Admin Panel",['Register','Login'],
                              menu_icon=['house'],icons=['lock','lock'],
                              default_index=0)

if selected == 'Register':
    st.header("Register Now")
    reg_email =st.text_input("Enter Username : ").strip()
    reg_password = st.text_input("Enter Password : ",type="password").strip()
    reg_button = st.button("Register")

    if reg_button:
        if reg_email and reg_password:
            if reg_email in df['Email'].values:
                st.error("User Already Exists, Please Login!")
            else:
                new_entry = pd.DataFrame({"Email":[reg_email],'Password':[reg_password]})
                df = pd.concat([df,new_entry],ignore_index=True)
                df.to_csv(user_data_file,index=False)
                st.success("Registration Successful, You can now Log In.")

        else:
            st.error("Please Enter Both Username and Password.")

if selected == 'Login':
    st.header("Login Now")
    email = st.text_input("Enter Username :").strip()
    password = st.text_input("Enter Password : ",type="password").strip()
    login_button = st.button("Login")

    if login_button:
        email =str(email).strip()
        password = str(password).strip()

        user =df[(df['Email'] == email) & (df['Password'] == password)]
        if not user.empty:
            st.session_state.authenticated = True
            st.session_state.user_id = email
            st.query_params.update(auth='true',user=email)
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Email or Password.")

if selected == 'About':
    st.title("🔮 Customer Segmentation")
    st.write("Customer Segmentation is a project aimed at analyzing customer data and segmenting customers into"
             "distinct groups based on various characteristics such as demographics, behaviour, and preferences. The "
             "goal is to uncover meaningful insights that can drive targeted marketing strategies, improve customer "
             "engagement, and enhance overall business performance.")

    st.subheader("🎯 Objectives:")

    st.write("1. Understand Customer Base: Gain insights into the diverse customer base by analyzing demographic information such as age, gender, income, and location.")
    st.write("2. Identify Customer Segments: Segment customers into distinct groups based on shared characteristics, including purchase behavior, frequency of purchases, product preferences, and engagement with marketing campaigns.")
    st.write("3. Personalized Marketing: Develop personalized marketing strategies tailored to different customer segments to improve campaign effectiveness and drive higher conversion rates.")
    st.write("4. Enhance Customer Experience: Identify opportunities to enhance the customer experience by delivering personalized recommendations, offers, and services based on individual preferences and behavior.")
    st.write("5. Optimize Resource Allocation: Allocate resources more effectively by focusing marketing efforts and resources on high-value customer segments with the greatest potential for revenue growth and profitability.")

    st.subheader("💡 Methodology:")

    st.write("1. Data Collection: Gather customer data from various sources, including transactional data, customer profiles, and marketing interactions.")
    st.write("2. Data Preprocessing: Cleanse and preprocess the data to ensure accuracy and consistency. Handle missing values, outliers, and data inconsistencies.")
    st.write("3. Exploratory Data Analysis (EDA): Perform exploratory data analysis to uncover patterns, trends, and relationships within the data. Visualize key metrics and distributions to gain a deeper understanding of the customer base.")
    st.write("4. Customer Segmentation: Utilize clustering algorithms such as K-means, hierarchical clustering, or Gaussian mixture models to segment customers into distinct groups based on predefined features.")
    st.write("5. Evaluation and Validation: Evaluate the quality of customer segments using appropriate metrics and validation techniques.")


if selected == 'Customer Wise Analysis':
    st.title("📊 Customer Wise Data Analysis")
    st.subheader("📌 DATASET")

    st.dataframe(data1.head(),hide_index=True)

    st.subheader("🗓️ YEAR WISE ANALYSIS")
    data1['Order Date'] = pd.to_datetime(data1['Order Date'], format='%d-%m-%Y')
    data1['Year'] = data1['Order Date'].dt.year

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    # Group and convert Year to string
    yearly_sales = data1.groupby('Year')['Sales'].sum().reset_index()
    yearly_sales['Year'] = yearly_sales['Year'].astype(str)
    st.dataframe(yearly_sales,hide_index=True)
    # st.write(yearly_sales.columns)
    st.subheader("📊 YEAR WISE SALES GRAPH")

    sales_graph = px.bar(yearly_sales, x='Year', y='Sales',
             labels={'Year': 'Year', 'Sales': 'Total Sales'},
             title='Yearly Sales Summary',
             color_discrete_sequence=['skyblue'])
    sales_graph.update_layout(xaxis=dict(type='category'))
    sales_graph.update_traces(marker_color=colors)
    st.plotly_chart(sales_graph,use_container_width=True)

    st.subheader("🏆 Top 10 HIGHEST SPENDERS")
    order_counts = data1['Customer ID'].value_counts().reset_index()
    order_counts.columns = ['Customer ID','Order ID']

    top_10_orders = order_counts.head(10)
    st.dataframe(top_10_orders,hide_index=True)

    fig = px.bar(top_10_orders,
                 x='Customer ID',
                 y='Order ID',
                 title='Top 10 Spenders',
                 labels={'Customer ID': 'Customer ID', 'Order ID': 'Number of Orders'},
                 color_continuous_scale='Blues')

    # Optional: sort bars in descending order
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("📉 Top 10 LOWEST SPENDERS")
    order_counts = data1['Customer ID'].value_counts().reset_index()
    order_counts.columns = ['Customer ID', 'Order ID']

    bottom_10_orders = order_counts.sort_values(by='Order ID',ascending=True).head(10)
    st.dataframe(bottom_10_orders,hide_index=True)

    fig = px.bar(bottom_10_orders,
                 x='Customer ID',
                 y='Order ID',
                 title='Bottom 10 Spenders',
                 labels={'Customer ID': 'Customer ID', 'Order ID': 'Number of Orders'},
                 color_continuous_scale='Reds')

    # Optional: sort bars in descending order
    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
    fig.update_traces(marker_color='#636EFA')
    st.plotly_chart(fig, use_container_width=True)


if selected == 'Region Wise Analysis':
    st.title("🌐 Region Wise Data Analysis")

    st.subheader("📊 STATE WISE SALES")

    state_sales = data1['State'].value_counts().reset_index()
    state_sales.columns = ['State','Sales']

    st.dataframe(state_sales)

    top_10_state = state_sales.head(10)

    state_graph = px.pie(top_10_state,names='State',values='Sales',title='Top 10 StateWise Sales',color='State',hole=0.3,labels='State')
    st.plotly_chart(state_graph,use_container_width=True)

    st.subheader("📈 Top 5 postal code by Sales")
    postal_codes = data1['Postal Code'].value_counts().reset_index()
    postal_codes.columns = ['Postal Code','Sales']
    top_5_postal = postal_codes.head()
    st.dataframe(top_5_postal,hide_index=True)
    postal_graph = px.bar(top_5_postal, x='Postal Code',y='Sales',
                          title='Top 5 Postal Codes',
                          labels={'Postal Code': 'Postal Code', 'Sales': 'Count'}
                          )
    postal_graph.update_layout(xaxis=dict(type='category'))
    postal_graph.update_traces(marker_color='#AB63FA')
    st.plotly_chart(postal_graph,use_container_width=True)

    st.subheader("📌 Region Wise Sale")

    region_sales = data1['Region'].value_counts().reset_index()
    region_sales.columns = ['Region','Count']

    st.dataframe(region_sales,hide_index=True)

    region_graph = px.pie(region_sales,names='Region',values='Count',title='Region-Wise Sales',color='Region',hole=0.3,labels='Region')
    st.plotly_chart(region_graph,use_container_width=True)

    st.subheader("📉 City-Wise Sales")

    city_sales = data1['City'].value_counts().reset_index()
    city_sales.columns = ['City','Order ID']

    st.dataframe(city_sales)
    city_sales_40 = city_sales.head(40).sort_values(by='City')

    city_graph = px.bar(city_sales_40,x='City',y='Order ID',title='Top 40 Cities Countsplot',
                        labels={'City': 'City', 'Order ID': 'Count'})
    city_graph.update_xaxes(tickangle=90)
    st.plotly_chart(city_graph,use_container_width=True)


if selected == 'Product Wise Analysis':
    st.title("📍 Product Wise Data Analysis")
    st.subheader("🔗 Category Wise Sales")

    category_sales = data1['Category'].value_counts().reset_index()
    category_sales.columns = ['Category','Order ID']

    st.dataframe(category_sales,hide_index=True)

    category_graph = px.pie(category_sales,names='Category',values='Order ID',title='Category-Wise Orders',color='Category',hole=0.3)
    st.plotly_chart(category_graph,use_container_width=True)


    st.subheader("🏆 Top 10 Highest Selling Product")

    product_sales = data1['Product Name'].value_counts().reset_index()
    product_sales.columns = ['Product Name','Count']

    top_10_products = product_sales.head(10)
    st.dataframe(top_10_products)

    product_graph = px.bar(top_10_products,x='Product Name',y='Count',title='Top Selling Products')
    st.plotly_chart(product_graph,use_container_width=True)

    st.subheader("🔎 Sub-Category Data")

    sub_sales = data1['Sub-Category'].value_counts().reset_index()
    sub_sales.columns = ['Sub-Category','Count']

    st.dataframe(sub_sales,hide_index=True)

    sub_graph = px.bar(sub_sales,x='Sub-Category',y='Count',title='Sub-Category Sales')
    sub_graph.update_traces(marker_color='teal')
    st.plotly_chart(sub_graph,use_container_width=True)


if selected == 'Mall Customers Analysis':
    st.title("📊 Mall Customers Analysis")

    age_hist = px.histogram(data2,x='Age',nbins=20,title='Distribution of Age')
    st.plotly_chart(age_hist,use_container_width=True)

    scatter_income = px.scatter(data2,
                     x='Annual Income (k$)',
                     y='Spending Score (1-100)',
                     title='Annual Income vs Spending Score',
                     labels={
                         'Annual Income (k$)': 'Annual Income (k$)',
                         'Spending Score (1-100)': 'Spending Score'
                     },
                     color='Gender')
    st.plotly_chart(scatter_income,use_container_width=True)

    gender_spending = data2.groupby('Gender')['Spending Score (1-100)'].mean().reset_index()
    # st.dataframe(gender_spending)
    gender_spending.columns=['Gender','Spending Score (1-100)']

    gender_spending_graph = px.bar(gender_spending,x='Gender',y='Spending Score (1-100)',title='Gender vs Spending Score',color='Gender')
    st.plotly_chart(gender_spending_graph,use_container_width=True)

    annual_income_customer = data2['Annual Income (k$)'].tail(10).reset_index()
    annual_income_customer.columns = ['CustomerID','Annual Income (k$)']

    annual_graph = px.bar(annual_income_customer,x='CustomerID',y='Annual Income (k$)',title='Top 10 Highest Income Customers')
    annual_graph.update_traces(marker_color='violet')

    st.plotly_chart(annual_graph,use_container_width=True)


    # Create age intervals (bins)
    bins = [10, 20, 30, 40, 50, 60, 70]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70']
    data2['Age Group'] = pd.cut(data2['Age'], bins=bins, labels=labels, right=True)

    # Calculate average spending score per age group
    age_group_avg = data2.groupby('Age Group')['Spending Score (1-100)'].mean().reset_index()

    # Plot bar chart
    age_group_avg_graph = px.bar(age_group_avg,
                 x='Age Group',
                 y='Spending Score (1-100)',
                 title='Average Spending Score by Age Group',
                 labels={'Spending Score (1-100)': 'Average Spending Score'})
    age_group_avg_graph.update_traces(marker_color='teal')
    st.plotly_chart(age_group_avg_graph,use_container_width=True)


if selected == 'Mall Customers Prediction':
    st.title("🔮 Mall Customers Segmentation")

    X = data2[['Annual Income (k$)', 'Spending Score (1-100)']]

    kmeans = KMeans(n_clusters=5, random_state=42)
    data2['Cluster'] = kmeans.fit_predict(X)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    fig = px.scatter(
        data2,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Cluster',
        title='Customer Segments (K-Means Clustering)',
        labels={'Cluster': 'Customer Cluster'}
    )

    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
    st.plotly_chart(fig,use_container_width=True)


    st.subheader("📝 Income Wise Spending Clusters")

    with st.form('Predict'):
        st.subheader("Insert Your Details")
        salary_input = st.number_input("Annual Income (k$)", value=50)
        score_input = st.number_input("Spending Score (1-100)", value=50)
        predict = st.form_submit_button("Predict Cluster")

        if predict:
            prediction = kmeans.predict([[salary_input, score_input]])[0]  # Get the cluster number directly
            st.success(f"🎉 You belong to Cluster {prediction}")

            if prediction == 0:
                st.success("Cluster 0 - Moderate Income, Moderate Spending")
            elif prediction == 1:
                st.success("Cluster 1 - High Income, High Spending")
            elif prediction == 2:
                st.success("Cluster 2 - Low Income, High Spending")
            elif prediction == 3:
                st.success("Cluster 3 - High Income, Low Spending")
            elif prediction == 4:
                st.success("Cluster 4 - Low Income, Low Spending")


if selected == 'Market Customers Analysis':
    st.title("📌 Market Customer Dataset")

    st.dataframe(data3)

    st.write("### Data Cleaning")
    st.write("Fill Minimum Payment with Mean")
    data3['MINIMUM_PAYMENTS'] = data3['MINIMUM_PAYMENTS'].fillna(data3['MINIMUM_PAYMENTS'].mean())
    st.dataframe(data3['MINIMUM_PAYMENTS'])
    st.write("Fill Credit Limit with Mean")
    data3['CREDIT_LIMIT'] = data3['CREDIT_LIMIT'].fillna(data3['CREDIT_LIMIT'].mean())
    st.dataframe(data3['CREDIT_LIMIT'])
    st.write("Dataset after null values by columns")
    st.dataframe(data3.isnull().sum())


if selected == 'Market Customers Prediction':
    st.title("🔮 Customers Prediction")
    st.subheader("📌 Raw Data")
    st.dataframe(data3)

    X = data3[['BALANCE', 'PURCHASES']]
    inertia = []
    k_values = list(range(1, 11))
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)

    # Elbow plot using Plotly Express
    elbow_df = pd.DataFrame({'k': k_values, 'Inertia': inertia})
    fig_elbow = px.line(
        elbow_df, x='k', y='Inertia',
        markers=True,
        title='Elbow Method for Optimal Number of Clusters',
        template='plotly_white'
    )
    fig_elbow.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia')
    st.plotly_chart(fig_elbow,use_container_width=True)

    cluster_slider = st.slider("Select the number of clusters",2,10,4)

    kmeans = KMeans(n_clusters=cluster_slider, random_state=42)
    data3['Cluster'] = kmeans.fit_predict(X)

    st.subheader("📤 Clustering Results")
    st.dataframe(data3[['BALANCE', 'PURCHASES','Cluster']])

    # Interactive scatter plot (no centroids)
    fig_scatter = px.scatter(
        data3, x='BALANCE', y='PURCHASES',
        color=data3['Cluster'].astype(str),
        title='KMeans Clustering',
        labels={'color': 'Cluster'},
        template='plotly_white'
    )

    st.plotly_chart(fig_scatter,use_container_width=True)

if selected == 'Contact':
    CSV_FILE = "form_submissions.csv"
    st.title("📬 Contact Us")

    # Contact form
    with st.form(key='contact_form'):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Send")

        if submit_button:
            if not name or not email or not message:
                st.warning("⚠️ Please fill out all fields.")
            else:
                # Check if file exists
                file_exists = os.path.isfile(CSV_FILE)

                # Write to CSV
                with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Timestamp", "Name", "Email", "Message"])
                    writer.writerow([datetime.now().isoformat(), name, email, message])

                st.success("✅ Message sent successfully! We’ll get back to you soon.")


if selected == 'Logout':
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.query_params.clear()
    st.success("Logout Successfully")
    st.rerun()

