import json
import unittest

from src.h3spark import utils

input_geom = {
    "type": "Polygon",
    "coordinates": [[(0, 0), (1, 1), (0, 1), (0, 0)]],
}


class UtilTests(unittest.TestCase):
    def test_to_h3_shape_str(self):

        str = json.dumps(input_geom)
        result = utils.to_h3_shape(str)

        self.assertEqual(json.dumps(result.__geo_interface__), json.dumps(input_geom))

    def test_to_h3_shape_list(self):
        result = utils.to_h3_shape(
            [(lat, lon) for (lon, lat) in input_geom["coordinates"][0]]
        )

        self.assertEqual(json.dumps(result.__geo_interface__), json.dumps(input_geom))

    def test_to_h3_shape_multi_list(self):

        coords = [
            [(lat, lon) for (lon, lat) in input_geom["coordinates"][0]],
            [(lon, lat) for (lon, lat) in input_geom["coordinates"][0]],
        ]
        result = utils.to_h3_shape(coords)

        self.assertEqual(
            json.dumps(result.__geo_interface__),
            json.dumps(
                {
                    "type": "MultiPolygon",
                    "coordinates": [[c] for c in coords[::-1]],
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
