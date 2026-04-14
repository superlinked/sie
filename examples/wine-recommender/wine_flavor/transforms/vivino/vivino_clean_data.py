
def unique_flavors(df): 
    all_unique_flavors = set()
    for _, wine_row in df.iterrows(): 
        for flavor_group in wine_row['wine_flavors']: 
            for key_word in (flavor_group.get('primary_keywords') or []): 
                if key_word.get('name'): 
                    all_unique_flavors.add(key_word['name']) 
            
            for key_word in (flavor_group.get('secondary_keywords') or []): 
                if key_word.get('name'): 
                    all_unique_flavors.add(key_word['name']) 

    return sorted(all_unique_flavors)
