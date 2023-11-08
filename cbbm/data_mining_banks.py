import requests
from bs4 import BeautifulSoup
import pandas as pd

# # Define the URL of the webpage
# url = "https://www.bcra.gob.ar/SistemasFinancierosYdePagos/Entidades_financieras.asp"

# # Send an HTTP GET request to the webpage
# response = requests.get(url)

# # Check if the request was successful
# if response.status_code == 200:
#     # Parse the HTML content of the page
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Find the select element with the class "form-control"
#     select_element = soup.find('select', class_='form-control')

#     # Loop through the options within the select element
#     for option in select_element.find_all('option'):
#         option_text = option.text
#         option_value = option.get('value')
#         print(f"Option Text: {option_text}, Option Value: {option_value}")
# else:
#     print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

# # Define the URL and headers
# url = "https://www.bcra.gob.ar/SistemasFinancierosYdePagos/Entidades_financieras.asp"
# headers = {
#     'User-Agent': 'Your User-Agent Here',
# }

# # Define the payload data
# data = {
#     'bco': '00017',  # Replace 'your_key' with the actual key name
# }

# # Send a POST request with the payload
# response = requests.post(url, headers=headers, data=data)

# # Check if the request was successful
# if response.status_code == 200:
#     # You can now parse the response content and extract the updated information
#     # For example, you can use BeautifulSoup as mentioned earlier

#     soup = BeautifulSoup(response.text, 'html.parser')
#     print(soup.text)
#     # Process the updated data

# else:
#     print(f"Failed to send the request. Status code: {response.status_code}")

# url_bank = f"https://www.bcra.gob.ar/SistemasFinancierosYdePagos/Entidades_financieras_situacion_deudores.asp?bco={data['bco']}&nom="
# # Send a POST request with the payload
# response = requests.get(url_bank)

# # Check if the request was successful
# if response.status_code == 200:
#     # You can now parse the response content and extract the updated information
#     # For example, you can use BeautifulSoup as mentioned earlier

#     soup = BeautifulSoup(response.text, 'html.parser')
#     print(soup.text)
#     # Process the updated data

# else:
#     print(f"Failed to send the request. Status code: {response.status_code}")

# Define the URL
data = {'bco': '00017'}  # Replace '00017' with the actual bank code
url = f"https://www.bcra.gob.ar/SistemasFinancierosYdePagos/Entidades_financieras_situacion_deudores.asp?bco={data['bco']}&nom="

# Send an HTTP GET request to the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table element with the specified class
    table = soup.find('table', class_='table-BCRA')

    # Extract and store the table content in a list
    table_data = []
    for i, row in enumerate(table.find_all('tr')):
        columns = row.find_all('td')
        row_data = [col.text.strip() for col in columns]
        if i == 0: 
            data_columns = row_data
            len_columns = len(data_columns)
            pass
        if row_data == data_columns:
            print("header columns")
            pass
        else:
            print(row_data)
            table_data.append(row_data)
    
    table_reshaped = [table_data[i:i+len_columns] for i in range(0, len(table_data), len_columns)]

    # Create a DataFrame from the table data
    df = pd.DataFrame(table_reshaped, columns=data_columns)
    print(df.head())

    # Optionally, set the header row as column names
    # header = df.iloc[0]
    # df = df[1:]
    # df.columns = header

    # Save the DataFrame to a CSV file
    csv_filename = "bank_data.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Table data saved to {csv_filename}")

else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
