from load_data import ckan, datasets

def fetch_dataset_metadata(ckan, dataset_name: str) -> dict:
    """Выгружает полные метаданные одного датасета."""
    return ckan.action.package_show(id=dataset_name)


a = []
for ds in datasets:
    a.append(fetch_dataset_metadata(ckan, ds))

print(a)