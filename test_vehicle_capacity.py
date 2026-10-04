#!/usr/bin/env python3
"""Offline adversarial regression tests for D0-key's KB-backed vehicle capacity check.

Run from the repository root: python3 test_vehicle_capacity.py -v
"""
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The live D1 path uses no embeddings. Do not initialize the unused provider module.
sys.modules.setdefault("bedrock_embed", types.ModuleType("bedrock_embed"))

from data_loader import QueryMeta, SandboxDB
from scoring import TravelPlanScorer


class FixtureDB(SandboxDB):
    def __init__(self, rows=None):
        super().__init__()
        self.rows = rows or {
            "Paris": [
                {"car_id": 1, "car_type": "Compact", "capacity": 4, "price_per_day": 40},
                {"car_id": 2, "car_type": "Compact", "capacity": 4, "price_per_day": 45},
                {"car_id": 3, "car_type": "SUV", "capacity": 5, "price_per_day": 60},
            ],
            "Rome": [
                {"car_id": 1, "car_type": "Compact", "capacity": 4, "price_per_day": 35},
                {"car_id": 2, "car_type": "SUV", "capacity": 5, "price_per_day": 55},
            ],
        }

    def get_cars(self, city):
        return pd.DataFrame(self.rows.get(city, []))


def meta(cities=("Paris",), people=3, required=5, car_required=True):
    return QueryMeta(
        query="capacity regression fixture", level="easy", person_num=people,
        days=1, req_flight={"arrival_city": list(cities)}, req_hotel={},
        req_attraction=None, req_car={"capacity": str(required)} if car_required else {},
        cities_count=len(cities),
    )


def car(city="Paris", car_type="SUV", **fields):
    return {"city": city, "car_type": car_type, **fields}


def plan_data(bookings=(), cities=("Paris",)):
    flights = [{"departure_city": a, "arrival_city": b} for a, b in zip(cities, cities[1:])]
    return {"is_feasible": True, "flights": flights, "hotels": [], "attractions": [],
            "cars": list(bookings), "cities": ["Paris"],
            "plan": {"day1": {"current_city": "Paris"}}}


class VehicleCapacityTests(unittest.TestCase):
    def setUp(self):
        self.db = FixtureDB()
        self.scorer = TravelPlanScorer(self.db)

    def score(self, bookings=(), task=None):
        task = task or meta()
        return self.scorer.score_d0_keyword(
            plan_data(bookings, task.req_flight.get("arrival_city", [])), task)

    def test_four_seat_car_cannot_claim_eight_to_pass_five_seat_request(self):
        self.assertEqual(self.score([car(car_type="Compact", capacity=8)]), 0.0)

    def test_capacity_comes_from_kb_even_when_agent_understates_it(self):
        self.assertEqual(self.score([car(capacity=1)]), 1.0)

    def test_shared_capacity_does_not_require_choosing_a_price_variant(self):
        self.assertEqual(self.db.get_car_capacity("Compact", "Paris"), 4)
        self.assertEqual(self.score([car(car_type="Compact")], meta(required=4)), 1.0)

    def test_no_car_requirement_does_not_apply_capacity(self):
        with patch.object(self.db, "get_car_capacity", side_effect=AssertionError("not applicable")):
            self.assertEqual(self.score([], meta(car_required=False)), 1.0)
            self.assertEqual(self.score([car(car_type="Compact")],
                                        meta(people=8, car_required=False)), 1.0)

    def test_party_size_remains_a_floor_when_explicit_request_is_lower(self):
        task = meta(people=5, required=3)
        self.assertEqual(self.score([car(car_type="Compact")], task), 0.0)
        self.assertEqual(self.score([car()], task), 1.0)

    def test_missing_booking_city_or_kb_match_fails(self):
        for bookings in [[], [car(city="")], [car(city="Atlantis", capacity=99)],
                         [car(car_type="Imaginary", capacity=99)]]:
            with self.subTest(bookings=bookings):
                self.assertEqual(self.score(bookings), 0.0)

    def test_required_car_with_no_stay_city_fails_closed(self):
        self.assertEqual(self.score([car()], meta(cities=())), 0.0)

    def test_each_required_city_needs_a_matching_sufficient_car(self):
        task = meta(cities=("Paris", "Rome"))
        self.assertEqual(self.score([car(), car(city="Rome")], task), 1.0)
        self.assertEqual(self.score([car()], task), 0.0)
        self.assertEqual(self.score([car(), car(city="Rome", car_type="Compact")], task), 0.0)

    def test_good_car_cannot_mask_bad_booking_in_same_city(self):
        self.assertEqual(self.score([car(), car(car_type="Compact", capacity=99)]), 0.0)

    def test_missing_city_is_not_inferred_from_day_or_claimed_capacity(self):
        payload = {"is_feasible": True, "plan": {"day1": {
            "current_city": "Paris", "car": {"car_type": "SUV", "capacity": 99}}}}
        extracted = self.scorer.extract_plan_data(payload)
        self.assertEqual(self.scorer.score_d0_keyword(extracted, meta()), 0.0)

    def test_existing_top_level_car_extractor_behavior_is_preserved(self):
        payload = {"is_feasible": True, "plan": {
            "day1": {"current_city": "Paris"}, "car": car()}}
        extracted = self.scorer.extract_plan_data(payload)
        self.assertEqual(extracted["cars"], [])
        self.assertEqual(self.scorer.score_d0_keyword(extracted, meta()), 0.0)

    def test_capacity_failure_does_not_add_gate_or_change_other_dimensions(self):
        payload = {"is_feasible": True, "plan": {"day1": {
            "current_city": "Paris", "car": car(car_type="Compact", capacity=8)}}}
        enough = self.scorer.score_query(copy.deepcopy(payload), meta(required=4))
        too_small = self.scorer.score_query(copy.deepcopy(payload), meta(required=5))
        self.assertEqual((enough.d0_keyword, too_small.d0_keyword), (1.0, 0.0))
        self.assertTrue(too_small.fully_valid)
        self.assertTrue(too_small.task_success)  # legacy gate-based success is separate
        for key, value in vars(enough).items():
            if key != "d0_keyword":
                self.assertEqual(value, getattr(too_small, key), key)


class AmbiguousCapacityTests(unittest.TestCase):
    def setUp(self):
        self.db = FixtureDB({"Paris": [
            {"car_id": 1, "car_type": "Van", "capacity": 4, "price_per_day": 40},
            {"car_id": 2, "car_type": "Van", "capacity": 7, "price_per_day": 70},
        ]})
        self.scorer = TravelPlanScorer(self.db)

    def score(self, **fields):
        return self.scorer.score_d0_keyword(plan_data([car(car_type="Van", **fields)]), meta())

    def test_conflicting_capacity_cannot_take_max_or_trust_claim(self):
        self.assertIsNone(self.db.get_car_capacity("Van", "Paris"))
        self.assertEqual(self.score(capacity=99), 0.0)

    def test_exact_price_can_disambiguate_but_nearest_price_cannot(self):
        self.assertEqual(self.score(price_per_day=70), 1.0)
        self.assertEqual(self.score(price_per_day=40, capacity=99), 0.0)
        self.assertEqual(self.score(price_per_day=999, capacity=99), 0.0)

    def test_car_id_can_disambiguate_and_unknown_id_fails(self):
        self.assertEqual(self.score(car_id="2"), 1.0)
        self.assertEqual(self.score(car_id=1, capacity=99), 0.0)
        self.assertEqual(self.score(car_id=999, price_per_day=70, capacity=99), 0.0)

    def test_equal_prices_with_conflicting_capacities_remain_ambiguous(self):
        self.db.rows["Paris"][0]["price_per_day"] = 70
        self.assertEqual(self.score(price_per_day=70), 0.0)
        self.assertEqual(self.score(car_id=2, price_per_day=70), 1.0)

    def test_missing_or_invalid_kb_capacity_fails_closed(self):
        for bad in [None, float("nan"), float("inf"), -1, 4.5, "invalid"]:
            with self.subTest(capacity=bad):
                self.db.rows["Paris"] = [
                    {"car_id": 1, "car_type": "Van", "capacity": bad, "price_per_day": 40}]
                self.assertEqual(self.score(capacity=99), 0.0)


if __name__ == "__main__":
    unittest.main()
