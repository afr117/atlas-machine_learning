#!/usr/bin/env python3
"""
Module to interact with the Swapi API to find sentient planets
"""
import requests


def sentientPlanets():
    """
    Returns a list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        species_list = data.get('results', [])

        for species in species_list:
            # Check if sentient in either classification or designation
            classification = species.get('classification', "").lower()
            designation = species.get('designation', "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get('homeworld')

                if homeworld_url:
                    # Fetch the planet details
                    planet_res = requests.get(homeworld_url)
                    if planet_res.status_code == 200:
                        planet_name = planet_res.json().get('name')
                        # Requirement check: Avoid duplicates if necessary
                        # though the output shows specific order
                        if planet_name not in planets:
                            planets.append(planet_name)

        url = data.get('next')

    return planets
