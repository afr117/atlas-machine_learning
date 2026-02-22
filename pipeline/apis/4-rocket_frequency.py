#!/usr/bin/env python3
"""
Script to display the number of launches per rocket
"""
import requests


def rocket_frequency():
    """
    Fetches launches and rockets to count and display launch frequency
    """
    # 1. Fetch all rockets to map IDs to Names
    r_url = "https://api.spacexdata.com/v4/rockets"
    r_res = requests.get(r_url)
    if r_res.status_code != 200:
        return
    # Dictionary mapping ID -> Name
    rocket_map = {r.get('id'): r.get('name') for r in r_res.json()}

    # 2. Fetch all launches
    l_url = "https://api.spacexdata.com/v4/launches"
    l_res = requests.get(l_url)
    if l_res.status_code != 200:
        return
    launches = l_res.json()

    # 3. Count launches per rocket name
    stats = {}
    for launch in launches:
        r_id = launch.get('rocket')
        name = rocket_map.get(r_id)
        if name:
            stats[name] = stats.get(name, 0) + 1

    # 4. Sort: Primary by count (descending), Secondary by name (ascending)
    # The '-' in -item[1] handles descending for integers
    sorted_stats = sorted(stats.items(), key=lambda item: (-item[1], item[0]))

    # 5. Print results
    for name, count in sorted_stats:
        print(f"{name}: {count}")


if __name__ == '__main__':
    rocket_frequency()
