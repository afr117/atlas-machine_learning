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
            reset_ts = int(response.headers.get('X-Ratelimit-Reset', 0))
            current_ts = int(time.time())
            # Calculate minutes remaining
            minutes = int((reset_ts - current_ts) / 60)
            print(f"Reset in {minutes} min")

        # Handle 404 Not Found
        elif response.status_code == 404:
            print("Not found")

        # Handle 200 OK
        elif response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')
            if location:
                print(location)
            # If location is null/empty, usually nothing is printed 
            # or the requirement expects specific output.

    except Exception:
        pass


if __name__ == '__main__':
    get_user_location()
