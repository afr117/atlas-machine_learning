
#!/usr/bin/env python3
"""
Script to display the first SpaceX launch with specific details
"""
import requests


def get_first_launch():
    """
    Fetches all launches, sorts by date, and prints the first one's details
    """
    # 1. Fetch all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(launches_url)
    if response.status_code != 200:
        return

    launches = response.json()
    # 2. Sort by date_unix to find the earliest
    # If dates are the same, python's sort is stable (keeps original order)
    launches.sort(key=lambda x: x.get('date_unix'))

    first_launch = launches[0]

    # 3. Get specific details
    launch_name = first_launch.get('name')
    date_local = first_launch.get('date_local')

    # 4. Fetch Rocket Name
    rocket_id = first_launch.get('rocket')
    rocket_res = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_name = rocket_res.json().get('name') if rocket_res.status_code == 200 else "Unknown"

    # 5. Fetch Launchpad Name and Locality
    pad_id = first_launch.get('launchpad')
    pad_res = requests.get(f"https://api.spacexdata.com/v4/launchpads/{pad_id}")
    if pad_res.status_code == 200:
        pad_json = pad_res.json()
        pad_name = pad_json.get('name')
        pad_locality = pad_json.get('locality')
    else:
        pad_name, pad_locality = "Unknown", "Unknown"

    # 6. Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    print(f"{launch_name} ({date_local}) {rocket_name} - {pad_name} ({pad_locality})")


if __name__ == '__main__':
    get_first_launch()

