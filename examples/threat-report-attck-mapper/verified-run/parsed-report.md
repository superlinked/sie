MFA PSA, Oh My!
February 03, 2022
### Key Takeaways
* As multi-factor authentication becomes a standard security practice, phish kits are evolving with the times to steal these tokens and bypass this trusted layer of security.
* Threat actors are using phish kits that leverage transparent reverse proxy, which enables them to man-in-the-middle (MitM) a browser session and steal credentials and session cookies in real-time.
* It is likely that more threat actors will turn to these MitM phish kits, making security increasingly difficult for defenders.
### Overview
Since the inclusion of the first password in the [Compatible Time-Sharing System](https://en.wikipedia.org/wiki/Password#History) at MIT in 1961, people have been cognizant of information security.
While multi-factor authentication (MFA) did not enter the scene until years later in 1986 with the first RSA tokens, it has recently seen widespread adoption in the consumer space.
According to MFA digital authenticator company Duo's annual [State of the Auth Report](https://duo.com/blog/the-2021-state-of-the-auth-report-2fa-climbs-password-managers-biometrics-trend) 78% of respondents have used two/multi-factor authentication (2FA/MFA) in 2021 compared to just 28% in 2017.
While many companies like Duo and RSA have helped make MFA more ubiquitous and user-friendly, threat actors have not been resting on their laurels, choosing to target MFA as well as looking for ways to bypass MFA with evolving phishing kits.
![Duo State of the Auth Report 2021 shows increase in MFA usage.](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.17.28%20AM.png)
*Figure 1. Duo State of the Auth Report 2021 shows increase in MFA usage.*
### MFA Phishing Kit Evolution
Phishing kits are software developed to aid threat actors in harvesting credentials and quickly capitalizing on them.
Often installed on a dedicated server owned by the threat actor or covertly installed on a compromised server owned by an unlucky individual, many of these kits can be purchased for [less than a cup of coffee](/us/blog/threat-insight/have-money-latte-then-you-too-can-buy-phish-kit "Have Money for a Latte.
Then You Too Can Buy a Phish Kit").
Proofpoint threat researchers see numerous MFA phishing kits ranging from simple open-source kits with human readable code and no-frills functionality to sophisticated kits utilizing numerous layers of obfuscation and built-in modules that allow for stealing usernames, passwords, MFA tokens, social security numbers and credit card numbers.
At their core these kits are using the same techniques for harvesting credentials as the traditional kits that steal only usernames and passwords.
![Simple phishing kit utilizing an open directory ](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.19.05%20AM.png)
*Figure 2. Simple phishing kit utilizing an open directory with credentials stored on the same webserver as the kit.*
In recent years, Proofpoint researchers have observed the emergence of a new type of kit that does not rely on recreating a target website.
Instead, these kits use a [transparent reverse proxy](https://en.wikipedia.org/wiki/Reverse_proxy#:~:text=In%20computer%20networks%2C%20a%20reverse,mainly%20used%20to%20balance%20load. to present the actual website to the victim.
Modern web pages are dynamic and change frequently.
Therefore, presenting the actual site instead of a facsimile greatly enhances the illusion an individual is logging in safely.
Another advantage of the reverse proxy is that it allows the threat actor to man-in-the-middle (MitM) a session and capture not only the usernames and passwords in real-time, but also the session cookie.
![MitM Transparent Reverse Proxy](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.19.29%20AM.png)
*Figure 3. MitM Transparent Reverse Proxy.*
The session cookie (see Figure 4) can then be used by the threat actor to gain access to the targeted account without the need for a username, password, or MFA token.
![Evilginx2 LinkedIn session cookie example](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.19.50%20AM.png)
*Figure 4. Evilginx2 LinkedIn session cookie example.*
Proofpoint researchers have noted a small increase in the use of these phish kits and anticipate greater adoption by threat actors as MFA forces them to adapt.
Specifically, Proofpoint has noted three transparent reverse proxy kits emerging on the scene.
### Reverse Proxy Phish Kits
Modlishka: A Polish security researcher Piotr Duszyński developed Modliska and released it in December 2018 on github.com.
This relatively simple tool allows phishing one site at a time, sports a command line interface, and provides the threat actor with a handy GUI to retrieve the credentials and session information (see Figure 5).
Modlishka also integrates Let’s Encrypt so it can make the fake domain landing page just a bit more believable by encrypting the traffic and providing the little padlock in the web bar.
While Modlishka may not be as advanced as the other two kits discussed later in this blog, it is still capable of harvesting a victim’s session even when tech like Duo’s push notification [authenticator is used](https://commons.lib.jmu.edu/cgi/viewcontent.cgi?article=1004&context=masters202029).
![Modliska graphical user interface (GUI).](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.20.07%20AM.png)
*Figure 5. Modliska graphical user interface (GUI).*
Muraena/Necrobrowser: Muraena/Necrobrowser is a two-part tool for phishing session cookies, credentials, and much more.
Created in 2019 by Giuseppe Trotta and Michele Orrù, Muraena runs server-side and uses a crawler to scan the target site to ensure it can properly rewrite all the traffic needed to not alert the victim.
Once the victim's credentials and session cookie has been harvested by the threat actor, they can deploy Necrobrowser.
Necrobrowser is a headless browser, which is a browser without a graphical user interface used for automation, that leverages the stolen session cookies to log into the target site and do things such as change passwords, disable Google Workspace notifications, dump emails, change SSH session keys in GitHub, and download all code repositories.
Evilginx2: Evilginx2 is an easy to use, transparent reverse proxy written in Golang by security researcher and developer Kuba Gretzky (Figure 6).
This is a popular tool for both red teams and threat actors as it is easy to setup and configure using its proprietary “phishlets.
Phishlets are [yaml](https://en.wikipedia.org/wiki/YAML) configuration files the engine uses to configure the proxy to the target site.
Utilizing these “phishlets,” you can configure the server to phish multiple brands at once.
Evilginx2 allows you to configure a custom subdomain and landing page URL for each as well.
The kit comes with several pre-installed “phishlets,” but more can be created and added easily.
Once a victim clicks on the malicious link, they are taken to a secure page with assets being displayed exactly how they are on the target site.
After they log in, the credentials, including MFA codes, and session cookie are sent to the server in real-time and the victim is either redirected to a different page or allowed to continue through to the page.
The threat actor is then able to use the stolen session cookie to log in as the victim where they can take multiple actions like changing the password, copying data, or pretending to be the victim.
![Evilginx2 command line console](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.21.39%20AM.png)
*Figure 6. Evilginx2 command line console.*
### Outlook
Most of these kits have been around for years so why the renewed interest.
In their [recent paper](https://catching-transparent-phish.github.io/catching_transparent_phish.pdf), researchers from Stony Brook University and Palo Alto Networks (Kondracki et al) took a deep dive into MitM phishing kits and discovered that MitM phishing pages represent an industry blind spot.
The researchers developed a machine learning tool called Phoca to scan suspected phishing pages and try to determine if they were using a transparent reverse proxy to MitM credentials.
They were able to identify over 1200 MitM phishing sites.
Of those 1200+ sites only 43.7% of domains and 18.9% of IP addresses appeared on popular blocklists like VirusTotal.
In addition to this, they found that standard phishing sites had a lifespan of just under 24 hours whereas MitM phishing sites lasted longer, and 15% had a lifespan greater than 20 days.
As recently as the end of January 2021, Proofpoint researchers identified a MitM reverse proxy site that was active for more than 72 hours.
![MitM O365 page.](/sites/default/files/inline-images/Screen%20Shot%202022-02-01%20at%2011.22.38%20AM.png)
*Figure 7. MitM O365 page.(domain redacted as without the full path it redirects to NSFW content)*
In May 2021, Google discussed in a [blog post](https://blog.google/technology/safety-security/a-simpler-and-safer-future-without-passwords/) that they would soon start to require MFA when logging into a Google account sometime in late 2021 or early 2022 (it would still be optional for Google Workspace customers).
We are now in 2022, the pandemic still rages, many workers are still working from home, and many may not return to the office.
As more companies follow Google’s lead and start requiring MFA, threat actors will rapidly move to solutions like these MitM kits.
They are easy to deploy, free to use, and have proven effective at evading detection.
The industry needs to prepare to deal with blind spots like these before they can evolve in new unexpected directions.