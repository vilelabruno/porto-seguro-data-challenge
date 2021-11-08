from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_app")

search = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE"]
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
locations = [geocode(s) for s in search]

