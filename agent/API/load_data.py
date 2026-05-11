from ckanapi import RemoteCKAN

ckan = RemoteCKAN('https://repository.nsedc.ru')

# ВАЖНО: добавить скобки () для вызова метода
 # мировая-торговля, население рф, банкирование, спонсорство_в_здравоохранении, гендерные-роли
response = ckan.action.package_search(
    q="*:*",              # все датасеты
    fq="tags:мировая-торговля",  # фильтр по тегу (замени на нужный)
    rows=50               # сколько вернуть (макс 1000)
)

tags = ckan.action.tag_list()


print(tags[:50])
print(f"Найдено датасетов: {response['count']}")
for ds in response['results']:
    print(f"  - {ds['name']}: {ds.get('title', '')[:80]}")

datasets = ['eurostat-ext-lt-introeu27-2020','emiss_62525', 'emiss_59975', 'emiss_51578', 'emiss_34027', 'emiss_33649']