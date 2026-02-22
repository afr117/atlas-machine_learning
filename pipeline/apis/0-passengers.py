#!/usr/bin/env python3
"""
Module to interact with the Swapi API
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of ships that can hold a given number of passengers
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        results = data.get('results', [])

        for ship in results:
            passengers = ship.get('passengers', "")
            # Remove commas for numbers like '1,000'
            passengers = passengers.replace(',', '')

            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:
                # This handles cases where passengers is 'n/a', 'unknown', etc.
                continue

        # Move to the next page of results
        url = data.get('next')

    return ships
    
