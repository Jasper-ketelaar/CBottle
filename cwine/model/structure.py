from datetime import date


class Wine:
    sku: str
    created: date
    grape: str
    wine_type: str
    sparkling: bool
    region: str
    country: str
    winery: str
    name: str
    year: int
    content: float

    def __init__(self, sku: str, created: date, grape: str, wine_type: str, sparkling: bool,
                 region: str, country: str, winery: str, name: str, year: int, content: float = 0.75):
        self.content = content
        self.year = year
        self.name = name
        self.winery = winery
        self.country = country
        self.region = region
        self.sparkling = sparkling
        self.wine_type = wine_type
        self.grape = grape
        self.created = created
        self.sku = sku

    @staticmethod
    def from_row(row: dict):
        return Wine(
            row['sku'], date.fromisoformat(row['created']), row['grape'], row['wine_type'], row['sparkling'] == 'True',
            row['region'], row['country'], row['winery'], row['name'], row['year']
        )

    def __hash__(self):
        return self.sku.__hash__()

    def __repr__(self):
        return f'Wine(sku={self.sku}, sparkling={self.sparkling}, ' \
               f'name={self.name}, year={self.year}, grape={self.grape}' \
               f')'


class Winery:

    def __init__(self, name: str):
        self.name = name
        self.wines = dict()

    def __iadd__(self, other):
        if isinstance(other, Wine):
            self.wines[other.sku] = other
        return self

    def __contains__(self, item):
        if isinstance(item, Wine):
            return item.winery in self.wines
        elif isinstance(item, str):
            return item in self.wines

        return False

    def __getitem__(self, item):
        if isinstance(item, Wine):
            return self.wines[item.sku]
        elif isinstance(item, str):
            return self.wines[item]
        raise KeyError

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f'Winery(name={self.name}, wine_count={self.count_wines()}, wines={self.wines.values()})'

    def count_wines(self):
        return len(self.wines)


class Region:

    def __init__(self, name: str):
        self.name = name
        self.wineries = dict()

    def __iadd__(self, other):
        if isinstance(other, Winery):
            self.wineries[other.name] = other
        return self

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.wineries
        elif isinstance(item, Winery):
            return item in self.wineries.values()

        return False

    def __getitem__(self, item):
        if isinstance(item, Winery):
            return self.wineries[item.name]
        elif isinstance(item, str):
            return self.wineries[item]

        raise KeyError

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f'Region(name={self.name}, wine_count={self.count_wines()}, wineries={self.wineries.values()})'

    def count_wines(self):
        total = 0
        for winery in self.wineries.values():
            total += winery.count_wines()
        return total


class Country:

    def __init__(self, name: str):
        self.name = name
        self.regions = dict()

    def __iadd__(self, other):
        if isinstance(other, Region):
            self.regions[other.name] = other
        return self

    def __contains__(self, item):
        if isinstance(item, (Region, str)):
            return item in self.regions
        return False

    def __getitem__(self, item):
        if isinstance(item, Region):
            return self.regions[item.name]
        elif isinstance(item, str):
            return self.regions[item]
        raise KeyError

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f'Country(name={self.name}, wine_count={self.count_wines()}, regions={self.regions.values()})'

    def count_wines(self):
        total = 0
        for region in self.regions.values():
            total += region.count_wines()
        return total

    def drop_regions_below(self, count):
        self.regions = {name: region for (name, region) in self.regions.items() if region.count_wines() >= count}
