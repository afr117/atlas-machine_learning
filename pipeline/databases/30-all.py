#!/usr/bin/env python3
"""
Module that contains the function list_all
"""


def list_all(mongo_collection):
    """
    Lists all documents in a collection
    Returns an empty list if no document in the collection
    """
    return list(mongo_collection.find())
