training_data = {
    "politics": [
        "Putin orders troops into pro-Russian regions of eastern Ukraine.",
        "The president decided not to go through with his speech.",
        "There is much uncertainty surrounding the coming elections.",
        "Democrats are engaged in a ‘new politics of evasion’",
    ],
    "sports": [
        "The soccer team lost.",
        "The team won by two against zero.",
        "I love all sport.",
        "The olympics were amazing.",
        "Yesterday, the tennis players wrapped up wimbledon.",
    ],
    "weather": [
        "It is going to be sunny outside.",
        "Heavy rainfall and wind during the afternoon.",
        "Clear skies in the morning, but mist in the evenening.",
        "It is cold during the winter.",
        "There is going to be a storm with heavy rainfall.",
    ],
}
training_data_multi_label = {
    "politics": [
        "Putin orders troops into pro-Russian regions of eastern Ukraine.",
        "The president decided not to go through with his speech.",
        "There is much uncertainty surrounding the coming elections.",
        "Democrats are engaged in a ‘new politics of evasion’",
        "The soccer team lost.",
        "The team won by two against zero.",
        "I love all sport.",
    ],
    "sports": [
        "Putin orders troops into pro-Russian regions of eastern Ukraine.",
        "The president decided not to go through with his speech.",
        "There is much uncertainty surrounding the coming elections.",
        "Democrats are engaged in a ‘new politics of evasion’",
        "The soccer team lost.",
        "The team won by two against zero.",
        "I love all sport.",
        "The olympics were amazing.",
        "Yesterday, the tennis players wrapped up wimbledon.",
    ],
    "weather": [
        "It is going to be sunny outside.",
        "Heavy rainfall and wind during the afternoon.",
        "Clear skies in the morning, but mist in the evenening.",
        "It is cold during the winter.",
        "There is going to be a storm with heavy rainfall.",
    ],
}
training_data_single_class = {"politics": training_data["politics"]}

validation_data = [
    "I am surely talking about politics.",
    "Sports is all you need.",
    "The weather is amazing and sunny and cloudy.",
]

training_data_spans = {
    "good_weather": [
        "Sunny day, perfect for outdoor fun!",
        "Mild temps, great for a leisurely stroll.",
        "Clear skies, ideal for a picnic.",
        "Refreshing drizzle, a calm start.",
        "Beautiful sunset against a clear sky.",
    ],
    "bad_weather": [
        "Scorching heat, feels like an oven.",
        "Heavy rain, causing flooding.",
        "Thick fog, dangerous for driving.",
        "Freezing temps, strong winds.",
        "Poor air quality, limit outdoor activities.",
    ],
}


validation_data_spans = ["The weather is amazing and sunny and cloudy.", "This is bad weather and low temperatures."]
