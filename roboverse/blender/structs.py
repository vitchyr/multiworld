RevoluteJoint = namedtuple('RevoluteJoint', ['object', 'location', 'orientation', 'low', 'high'])
FixedJoint = namedtuple('FixedJoint', ['object', 'target'])
StationaryJoint = namedtuple('StationaryJoint', ['object'])