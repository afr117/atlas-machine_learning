#!/usr/bin/env python3
"""
Script to print the location of a specific GitHub user
"""
import requests
import sys
import time


def get_user_location():
    """
    Fetches and prints the location of a GitHub user from a provided URL
    """
    if len(sys.argv) < 2:
        return

    user_url = sys.argv[1]

    try:
        response = requests.get(user_url)

        # Handle 403 Forbidden (Rate Limit exceeded)
        if response.status_code == 403:
            reset_timestamp = int(response.headers.get('X-Ratelimit-Reset', 0))
            current_timestamp = int(time.time())
            # Calculate difference in minutes
            seconds_to_reset = reset_timestamp - current_timestamp
            minutes_to_reset = int(seconds_to_reset / 60)
            print(f"Reset in {minutes_to_reset} min")

        # Handle 404 Not Found
        elif response.status_code == 404:
            print("Not found")

        # Handle 200 OK
        elif response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')
            if location:
                print(location)
            else:
                print("No location found")

    except requests.exceptions.RequestException:
        pass


if __name__ == '__main__':
    get_user_location()
