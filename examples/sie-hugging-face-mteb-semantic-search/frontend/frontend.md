# Frontend for LLM Search

## Search Menu
Two menu points: 
1. Download LLM Cards
2. Browse LLM Cards

### Browse LLM Cards
This are two editable fields above:
Storage ID: Default "test01"
LLM ID: This together with Storage ID functions as a search possiblity for LLM Cards

Then comes an updated list of LLM Cards with ID, short description and other relevant fields in max 2-3 lines

#### Browser LLM Card Details Full
If user clicks on a short LLM Card, then this LLM Card Detail screen starts with all the data we have on that LLM id (in this storage ID)
There is a back button to return to the Browse LLM Cards screen

#### Generate Descriptions popup
The following UI elements are there
- text area with pormpt to create max 6K description
- button "Generate 6K description"
- button "Generate 2K description"
- button "Generate 200 char description"
- text area 6K description
- text area 2K description
- text area 200 description
##### Functionality Button Generate 6K Description
The prompt text area has already variables replaced. So if the button is pressed, an openrouter model is called with the prompt and the results are filled into the 6K description. This field is not saved.
##### Functionality Button Generate 2K Description
Based on the 6K description a 2K description is generated and filled into the corresponding text area. It is not yet saved.
The corresponding prompt is used for this (find good prompt even change prompt if needed)
##### Functionality Button Generate 200 char Description
Based on the 6K description a 200 character description is generated and filled into the corresponding text area. It is not yet saved.
##### Functionality Button Save Descriptions
2K and 200 char descriptions are filled and saved into sqlite table Models. 6K description is not saved