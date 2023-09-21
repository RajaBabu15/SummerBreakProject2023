from bs4 import BeautifulSoup

def extract_table_data(file_name):
    with open(file_name, 'r') as file:
        data = file.read()

    soup = BeautifulSoup(data, 'html.parser')
    table = soup.find('table')

    result = []
    for row in table.find_all('tr'):
        columns = row.find_all('td')
        result.append('\t'.join(column.text for column in columns))

    return '\n'.join(result)

print(extract_table_data('Testoing\sample.txt'))
