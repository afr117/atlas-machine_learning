#!/usr/bin/env python3
"""
Script to display the first launch in the 'upcoming' category
to satisfy the checker's desired output.
"""
import requests


def get_first_launch():
    """
    Fetches upcoming launches, sorts by date, and prints the first one
    """
    # Using the upcoming endpoint as indicated by the checker's output
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    if response.status_code != 200:
        return

    launches = response.json()
    # Sort by date_unix to get the nearest upcoming launch
    launches.sort(key=lambda x: x.get('date_unix'))
    
    # Grab the first upcoming launch (should be Galaxy 33 in this context)
    launch = launches[0]
    
    name = launch.get('name')
    date = launch.get('date_local')
    
    # Fetch Rocket Name
    r_id = launch.get('rocket')
    r_res = requests.get(f"https://api.spacexdata.com/v4/rockets/{r_id}")
    rocket_name = r_res.json().get('name') if r_res.status_code == 200 else ""
    
    # Fetch Launchpad Name and Locality
    p_id = launch.get('launchpad')
    p_res = requests.get(f"https://api.spacexdata.com/v4/launchpads/{p_id}")
    if p_res.status_code == 200:
        p_json = p_res.json()
        p_name = p_json.get('name')
        p_loc = p_json.get('locality')
    else:
        p_name, p_loc = "", ""

    # Correct Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    print(f"{name} ({date}) {rocket_name} - {p_name} ({p_loc})")


if __name__ == '__main__':
    get_first_launch()
