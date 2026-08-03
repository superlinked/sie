# Source and data notes

The example combines public government documents, a public-domain photograph,
and an explicitly fictional claim.

## FEMA NFIP Proof of Loss

- File: https://www.fema.gov/sites/default/files/2020-07/FEMA-Form_086-0-09_proof-of-loss.pdf
- Form: FEMA Form 086-0-09 (04/17)
- Use: the example fills the official form with fictional claim data
- Rights: U.S. federal government work under 17 U.S.C. § 105
- Reproducibility: the exact government PDF is bundled as a fallback because
  FEMA may reject requests from datacenter IP addresses

## Standard Flood Insurance Policy, Dwelling Form

- File: https://www.fema.gov/sites/default/files/documents/fema_F-122-Dwelling-SFIP_2021.pdf
- Use: policy retrieval and cited claim-review findings
- Rights: U.S. federal government work under 17 U.S.C. § 105
- Reproducibility: the exact government PDF is bundled as a fallback because
  FEMA may reject requests from datacenter IP addresses

## Flooded house interior photograph

- File page: https://commons.wikimedia.org/wiki/File:Flooded_house_interior.jpg
- Author: U.S. Fish and Wildlife Service
- Use: visual inspection of the submitted damage photograph
- Rights: public domain U.S. federal government work
- Reproducibility: the exact photograph is bundled as a fallback for Wikimedia
  rate limits

## Fictional claim

`claim.json` and `claim-note.txt` contain no real policyholder information.
`prepare-claim` fills the FEMA form and produces the attached estimate. The
deliberate evidence issues are an unsigned proof of loss and a $400 difference
between the form and the estimate.
