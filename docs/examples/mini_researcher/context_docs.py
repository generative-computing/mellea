# pytest: ollama, llm

from dataclasses import dataclass

from docs.examples.mini_researcher import RAGDocument

hashi = """HashiCorp, Inc. is an American software company[2] with a freemium business model based in San Francisco, California. HashiCorp provides tools and products that enable developers, operators and security professionals to provision, secure, run and connect cloud-computing infrastructure.[3] It was founded in 2012 by Mitchell Hashimoto and Armon Dadgar.[4][5] The company name HashiCorp is a portmanteau of co-founder last name Hashimoto and Corporation.[6]

HashiCorp is headquartered in San Francisco, but their employees are distributed across the United States, Canada, Australia, India, and Europe. HashiCorp offers source-available libraries and other proprietary products.[7][8]

History

Founders Armon Dadgar and Mitchell Hashimoto
HashiCorp was founded in 2012 by two classmates from the University of Washington, Mitchell Hashimoto and Armon Dadgar.[9] Co-founder Hashimoto was previously working on open-source software called Vagrant, which became incorporated into HashiCorp.[10]

On 29 November 2021, HashiCorp set terms for its IPO at 15.3 million shares at $68-$72 at a valuation of $13 billion.[11] It offered 15.3 million shares.[12] HashiCorp considers its workers to be remote workers first rather than coming into an office on a full-time basis.[13]

Around April 2021, a supply chain attack using code auditing tool codecov allowed hackers limited access to HashiCorp's customers networks.[14] As a result, private credentials were leaked. HashiCorp revoked a private signing key and asked its customers to use a new rotated key.

Mitchell Hashimoto resigned from the company in December 2023.[15]

Acquisition by IBM
On April 24, 2024, the company announced it had entered into an agreement to be acquired by IBM for $6.4 billion, with the transaction expected to close by the end of the same year.[16] This led to the Competition and Markets Authority of the United Kingdom launching an investigation into the acquisition in late 2024.[17][18] The deal closed on February 27, 2025 for $6.4 billion after receiving the necessary regulatory approvals.[19][20]

Products
HashiCorp provides a suite of tools intended to support the development and deployment of large-scale service-oriented software installations. Each tool is aimed at specific stages in the life cycle of a software application, with a focus on automation. Many have a plugin-oriented architecture in order to provide integration with third-party technologies and services.[21] Additional proprietary features for some of these tools are offered commercially and are aimed at enterprise customers.[22]

The main product line consists of the following tools:[3][21]

Vagrant (first released in 2010[23]): supports the building and maintenance of reproducible software-development environments via virtualization technology.
Packer [Wikidata] (first released in June 2013[24][25]): a tool for building virtual-machine images for later deployment.
Terraform (first released in July 2014): infrastructure as code software which enables provisioning and adapting virtual infrastructure across all major cloud providers.
Consul (first released in April 2014[26][21]): provides service mesh, DNS-based service discovery, distributed KV storage, RPC, and event propagation. The underlying event, membership, and failure-detection mechanisms are provided by Serf, an open-source library also published by HashiCorp.
Vault (first released in April 2015[27]): provides secrets management, identity-based access, encrypting application data and auditing of secrets for applications, systems, and users.[22]
Nomad (released in September 2015[28]): supports scheduling and deployment of tasks across worker nodes in a cluster.
Serf (first released in 2013): a decentralized cluster membership, failure detection, and orchestration software product.[29]
Sentinel (first released in 2017[30][31]): a policy as code framework for HashiCorp products.[32]
Boundary (first released in October 2020[33]): provides secure remote access to systems based on trusted identity.
Waypoint (first released in October 2020[34]): provides a modern workflow to build, deploy, and release across platforms.
"""  # codespell:ignore hashi

redhat = """Red Hat, Inc. (formerly Red Hat Software, Inc.) is an American software company that provides open source software products to enterprises[7] and is a subsidiary of IBM. Founded in 1993, Red Hat has its corporate headquarters in Raleigh, North Carolina, with other offices worldwide.

Red Hat has become associated to a large extent with its enterprise operating system Red Hat Enterprise Linux. With the acquisition of open-source enterprise middleware vendor JBoss, Red Hat also offers Red Hat Virtualization (RHV), an enterprise virtualization product. Red Hat provides storage, operating system platforms, middleware, applications, management products, support, training, and consulting services.

Red Hat creates, maintains, and contributes to many free software projects. It has acquired the codebases of several proprietary software products through corporate mergers and acquisitions, and has released such software under open source licenses. As of March 2016, Red Hat is the second largest corporate contributor to the Linux kernel version 4.14 after Intel.[8]

On October 28, 2018, IBM announced its intent to acquire Red Hat for $34 billion.[9][10][11] The acquisition closed on July 9, 2019.[12] It now operates as an independent subsidiary.[13][12]

History
In 1993, Bob Young incorporated the ACC Corporation, a catalog business that sold Linux and Unix software accessories. In 1994, Marc Ewing created his own Linux distribution, which he named Red Hat Linux[14] (associated with the time Ewing wore a red Cornell University lacrosse hat, given to him by his grandfather, while attending Carnegie Mellon University[15][16][17]). Ewing released the software in October, and it became known as the Halloween release. Young bought Ewing's business in 1995,[clarification needed] and the two merged to become Red Hat Software, with Young serving as chief executive officer (CEO).

Red Hat went public on August 11, 1999, achieving—at the time—the eighth-biggest first-day gain in the history of Wall Street.[14] Matthew Szulik succeeded Bob Young as CEO in December of that year.[18] Bob Young went on to found the online print on demand and self-publishing company, Lulu in 2002.

On November 15, 1999, Red Hat acquired Cygnus Solutions. Cygnus provided commercial support for free software and housed maintainers of GNU software products such as the GNU Debugger and GNU Binutils. One of the founders of Cygnus, Michael Tiemann, became the chief technical officer of Red Hat and by 2008 the vice president of open-source affairs. Later Red Hat acquired WireSpeed, C2Net, Hell's Kitchen Systems, and Akopia.[19][20]

In February 2000, InfoWorld awarded Red Hat its fourth consecutive "Operating System Product of the Year" award for Red Hat Linux 6.1.[21] Red Hat acquired Planning Technologies, Inc. in 2001 and AOL's iPlanet directory and certificate-server software in 2004.

Red Hat moved its headquarters from Durham to North Carolina State University's Centennial Campus in Raleigh, North Carolina in February 2002. In the following month Red Hat introduced Red Hat Linux Advanced Server,[22][23] later renamed Red Hat Enterprise Linux (RHEL). Dell,[24] IBM,[25] HP[26] and Oracle Corporation[27] announced their support of the platform.[28]

In December 2005, CIO Insight magazine conducted its annual "Vendor Value Survey", in which Red Hat ranked #1 in value for the second year in a row.[29] Red Hat stock became part of the NASDAQ-100 on December 19, 2005.

Red Hat acquired open-source middleware provider JBoss on June 5, 2006, and JBoss became a division of Red Hat. On September 18, 2006, Red Hat released the Red Hat Application Stack, which integrated the JBoss technology and which was certified by other well-known software vendors.[30][31] On December 12, 2006, Red Hat stock moved from trading on NASDAQ (RHAT) to the New York Stock Exchange (RHT). In 2007 Red Hat acquired MetaMatrix and made an agreement with Exadel to distribute its software.

On March 15, 2007, Red Hat released Red Hat Enterprise Linux 5, and in June acquired Mobicents. On March 13, 2008, Red Hat acquired Amentra, a provider of systems integration services for service-oriented architecture, business process management, systems development, and enterprise data services.

On July 27, 2009, Red Hat replaced CIT Group in Standard and Poor's 500 stock index, a diversified index of 500 leading companies of the U.S. economy.[32][33] This was reported as a major milestone for Linux.[34][35]

On December 15, 2009, it was reported that Red Hat will pay US$8.8 million to settle a class action lawsuit related to the restatement of financial results from July 2004. The suit had been pending in the U.S. District Court for the Eastern District of North Carolina. Red Hat reached the proposed settlement agreement and recorded a one-time charge of US$8.8 million for the quarter that ended Nov. 30.[36]

On January 10, 2011, Red Hat announced that it would expand its headquarters in two phases, adding 540 employees to the Raleigh operation, and investing over US$109 million. The state of North Carolina is offering up to US$15 million in incentives. The second phase involves "expansion into new technologies such as software virtualization and technology cloud offerings".[37]



Red Hat Tower with earlier company logo
On August 25, 2011, Red Hat announced it would move about 600 employees from the N.C. State Centennial Campus to the Two Progress Plaza building.[38] A ribbon cutting ceremony was held on June 24, 2013, in the re-branded Red Hat Headquarters.[39]

In 2012, Red Hat became the first one-billion dollar open-source company, reaching US$1.13 billion in annual revenue during its fiscal year.[40] Red Hat passed the $2 billion benchmark in 2015. As of February 2018 the company's annual revenue was nearly $3 billion.[41]

On October 16, 2015, Red Hat announced its acquisition of IT automation startup Ansible, rumored for an estimated US$100 million.[42]

In June 2017, Red Hat announced Red Hat Hyperconverged Infrastructure (RHHI) 1.0 software product.[43]

In May 2018, Red Hat acquired CoreOS.[44]

Red Hat's links to Israel's military and professed support for Israel have also led to some controversy and calls for boycott during the ongoing conflict in Gaza.[45][46][47]

IBM subsidiary
On October 28, 2018, IBM announced its intent to acquire Red Hat for US$34 billion, in one of its largest-ever acquisitions. The company will operate out of IBM's Hybrid Cloud division.[48][49]

Six months later, on May 3, 2019, the US Department of Justice concluded its review of IBM's proposed Red Hat acquisition,[50] and according to Steven J. Vaughan-Nichols "essentially approved the IBM/Red Hat deal".[51] The acquisition was closed on July 9, 2019.[52]

Fedora Project

Fedora Project logo
Main article: Fedora Project
Red Hat is the primary sponsor of the Fedora Project, a community-supported free software project that aims to promote the rapid progress of free and open-source software and content.[53]

Business model
Red Hat operates on a business model based on open-source software, development within a community, professional quality assurance, and subscription-based customer support. They produce open-source code so that more programmers can make adaptations and improvements.

Red Hat sells subscriptions for the support, training, and integration services that help customers in using their open-source software products. Customers pay one set price for unlimited access to services such as Red Hat Network and up to 24/7 support.[54]

In September 2014, however, CEO Jim Whitehurst announced that Red Hat was "in the midst of a major shift from client-server to cloud-mobile".[55]

Rich Bynum, a member of Red Hat's legal team, attributes Linux's success and rapid development partially to open-source business models, including Red Hat's.[56]

Programs and projects

This article needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed.
Find sources: "Red Hat" – news · newspapers · books · scholar · JSTOR (January 2025) (Learn how and when to remove this message)

Red Hat Summit is an annual conference, here seen in 2019.
One Laptop per Child
Red Hat engineers worked with the One Laptop per Child initiative (a non-profit organization established by members of the MIT Media Lab) to design and produce an inexpensive laptop and try to provide every child in the world with access to open communication, open knowledge, and open learning. The XO-4 laptop, the last machine the project produced (in 2012), runs a slimmed-down version of Fedora 17 as its operating system.

KVM
Avi Kivity began the development of KVM in mid-2006 at Qumranet, a technology startup company that was acquired by Red Hat in 2008.[57][58][59]

GNOME
Red Hat is the largest contributor to the GNOME desktop environment. It has several employees working full-time on Evolution, the official personal information manager for GNOME.

systemd
Init system and system/service manager for Linux systems.

PulseAudio
Network-capable sound server program distributed via the freedesktop.org project.

Dogtail
Dogtail, an open-source automated graphical user interface (GUI) test framework initially developed by Red Hat, consists of free software released under the GNU General Public License (GPL) and is written in Python. It allows developers to build and test their applications. Red Hat announced the release of Dogtail at the 2006 Red Hat Summit.[60][61]

MRG
Red Hat MRG is a clustering product intended for integrated high-performance computing (HPC). The acronym MRG stands for "Messaging Realtime Grid".

Red Hat Enterprise MRG replaces the kernel of Red Hat Enterprise Linux RHEL, a Linux distribution developed by Red Hat, to provide extra support for real-time computing, together with middleware support for message brokerage and scheduling workload to local or remote virtual machines, grid computing, and cloud computing.[62]

As of 2011, Red Hat works with the Condor High-Throughput Computing System community and also provides support for the software.[63]

The Tuna performance-monitoring tool runs in the MRG environment.[64]

Opensource.com
Red Hat produced the online publication Opensource.com since January 20, 2010.[65] The site highlights ways open-source principles apply in domains other than software development. The site tracks the application of open-source philosophy to business, education, government, law, health, and life.

The company originally produced a newsletter called Under the Brim. Wide Open magazine first appeared in March 2004, as a means for Red Hat to share technical content with subscribers regularly. The Under the Brim newsletter and Wide Open magazine merged in November 2004, to become Red Hat Magazine. In January 2010, Red Hat Magazine became Opensource.com.[66] In April 2023 Red Hat went through company layoffs and laid off the team maintaining Opensource.com.[67]

Red Hat Exchange
In 2007, Red Hat announced that it had reached an agreement with some free software and open-source (FOSS) companies that allowed it to make a distribution portal called Red Hat Exchange, reselling FOSS software with the original branding intact.[68][69] However, by 2010, Red Hat had abandoned the Exchange program to focus their efforts more on their Open Source Channel Alliance which began in April 2009.[70]

Red Hat build of Keycloak
Red Hat build of Keycloak[71] (formerly known as Red Hat Single Sign-On[72]) is a software product to allow single sign-on with Identity Management and Access Management aimed at modern applications and services. It is based on the open-source project Keycloak, which acts as an upstream project.

Red Hat Subscription Management
Red Hat Subscription Management (RHSM)[73] combines content delivery with subscription management.[74]

Ceph Storage
Red Hat is the largest contributor to the Ceph Storage SDS project : Block, File & Object Storage which runs on industry-standard x86 servers and Ethernet IP as well as ARM, InfiniBand, and other technologies.

Ceph aims primarily for completely distributed operation without a single point of failure, scalable to the exabyte level.

Ceph replicates data and makes it fault-tolerant, using commodity hardware and requiring no specific hardware support. Ceph's system offers disaster recovery and data redundancy through techniques such as replication, erasure coding, snapshots and storage cloning. As a result of its design, the system is both self-healing and self-managing, aiming to minimize administration time and other costs.

In this way, administrators have a single, consolidated system that avoids silos and collects the storage within a common management framework. Ceph consolidates several storage use cases and improves resource utilization. It also lets an organization deploy servers where needed.

OpenShift
Red Hat operates OpenShift, a cloud computing platform as a service, supporting applications written in Node.js, PHP, Perl, Python, Ruby, JavaEE and more.[75]

On July 31, 2018, Red Hat announced the release of Istio 1.0, a microservices management program used in tandem with the Kubernetes platform. The software purports to provide "traffic management, service identity and security, policy enforcement and telemetry" services in order to streamline Kubernetes use under the various Fedora-based operating systems. Red Hat's Brian Redbeard Harring described Istio as "aiming to be a control plane, similar to the Kubernetes control plane, for configuring a series of proxy servers that get injected between application components".[76] Also Red Hat is the second largest contributor to Kubernetes code itself, after Google.[77]

OpenStack
Red Hat markets a version of OpenStack which helps manage a data center in the manner of cloud computing.[78]

CloudForms
Red Hat CloudForms provides management of virtual machines, instances and containers based on VMware vSphere, Red Hat Virtualization, Microsoft Hyper-V, OpenStack, Amazon EC2, Google Cloud Platform, Microsoft Azure, and Red Hat OpenShift. CloudForms is based on the ManageIQ project that Red Hat open sourced. Code in ManageIQ is from the over US$100 million acquisition of ManageIQ in 2012.[79][80]

CoreOS
Container Linux (formerly CoreOS Linux) is a discontinued open-source lightweight operating system based on the Linux kernel and designed for providing infrastructure to clustered deployments. As an operating system, Container Linux provided only the minimal functionality required for deploying applications inside software containers, together with built-in mechanisms for service discovery and configuration sharing.

LibreOffice
Red Hat contributed, with several software developers, to LibreOffice, a free and open-source office suite.[81] However, in 2023, Red Hat announced they were not going to include LibreOffice in RHEL 10, citing the ability to download LibreOffice from Flatpak on RHEL desktops.[82]

Other FOSS projects
Red Hat also organises "Open Source Day" events[83] where multiple partners show their open-source technologies.[84]

Xorg
Red Hat is one of the largest contributors to the X Window System.[85][86]

Utilities and tools
Subscribers have access to:

Red Hat Developer Toolset (DTS)[87] – performance analysis and development tools[88]
Red Hat Software Collections (RHSCL) [89]
Over and above Red Hat's major products and acquisitions, Red Hat programmers have produced software programming-tools and utilities to supplement standard Unix and Linux software. Some of these Red Hat "products" have found their way from specifically Red Hat operating environments via open-source channels to a wider community. Such utilities include:

Disk Druid – for disk partitioning[90]
rpm – for package management
sos (son of sysreport) – tools for collecting information on system hardware and configuration.[91]
sosreport – reports system hardware and configuration details[92][citation needed]
SystemTap – tracing tool for Linux kernels, developed with IBM, Hitachi, Oracle[93] and Intel[94]
NetworkManager
The Red Hat website lists the organization's major involvements in free and open-source software projects.[95]

Community projects under the aegis of Red Hat include:

the Pulp application for software repository management.[96]
Subsidiaries
Red Hat Czech
Red Hat Czech


Red Hat building in Brno
Company type	Společnost s ručením omezeným (Limited Liability Company)
Industry	Software
Predecessor	Container Linux
Cygnus Solutions
Founded	2006; 19 years ago
Headquarters	Brno, Czech Republic
Revenue
Increase CZK 1,002 million (FY 2016)
CZK 806 million (FY 2015)
Net income
Increase CZK 123 million (FY 2016)
CZK 39 million (FY 2015)
Total assets
Increase CZK 420 million (FY 2016)
CZK 325 million (FY 2015)
Number of employees	1180 (2019)
Parent	Red Hat
Website	redhat.com/en/global/czech-republic
Footnotes / references
[97]
Red Hat Czech s.r.o. is a research and development arm of Red Hat, based in Brno, Czech Republic.[97] The subsidiary was formed in 2006 and has 1,180 employees (2019).[98] Red Hat chose to enter the Czech Republic in 2006 over other locations due to the country's embrace of open-source.[99] The subsidiary expanded in 2017 to a second location in the Brno Technology Park to accommodate an additional 350 employees.[100]

In 2016, Red Hat Czech reported revenue of CZK 1,002 million (FY 2016), and net income of CZK 123 million (FY 2016), with assets of CZK 420 million (FY 2016)|CZK 325 million (FY 2015).

The group was named the "Most progressive employer of the year" in the Czech Republic in 2010,[101] and the "Best Employer in the Czech Republic" for large scale companies in 2011 by Aon.[102]

Red Hat India
In 2000, Red Hat created the subsidiary Red Hat India to deliver Red Hat software, support, and services to Indian customers.[103] Colin Tenwick, former vice president and general manager of Red Hat EMEA, said Red Hat India was opened "in response to the rapid adoption of Red Hat Linux in the subcontinent. Demand for open-source solutions from the Indian markets is rising and Red Hat wants to play a major role in this region."[103] Red Hat India has worked with local companies to enable the adoption of open-source technology in both government[104] and education.[105]

In 2006, Red Hat India had a distribution network of more than 70 channel partners spanning 27 cities across India.[106] Red Hat India's channel partners included MarkCraft Solutions, Ashtech Infotech Pvt Ltd., Efensys Technologies, Embee Software, Allied Digital Services, and Softcell Technologies. Distributors include Integra Micro Systems[107] and Ingram Micro.

Mergers and acquisitions
Red Hat's first major acquisition involved Delix Computer GmbH-Linux Div, the Linux-based operating-system division of Delix Computer, a German computer company, on July 30, 1999.

Red Hat acquired Cygnus Solutions, a company that provided commercial support for free software, on January 11, 2000 – it was the company's largest acquisition, for US$674 million.[108] Michael Tiemann, co-founder of Cygnus, served as the chief technical officer of Red Hat after the acquisition. Red Hat made the most acquisitions in 2000 with five: Cygnus Solutions, Bluecurve, Wirespeed Communications, Hell's Kitchen Systems, and C2Net. On June 5, 2006, Red Hat acquired open-source middleware provider JBoss for US$420 million and integrated it as its own division of Red Hat.

On December 14, 1998, Red Hat made its first divestment, when Intel and Netscape acquired undisclosed minority stakes in the company. The next year, on March 9, 1999, Compaq, IBM, Dell and Novell each acquired undisclosed minority stakes in Red Hat.

Acquisitions
Date	Company	Business	Country	Value (USD)	References
July 13, 1999	Atomic Vision	Website design	 United States	—	[109][110]
July 30, 1999	Delix Computer GmbH
-Linux Div[note 1]	Computers and software	 Germany	—	[111]
January 11, 2000	Cygnus Solutions Limited	gcc, gdb, binutils	 United States	$674,444,000	[112][108]
May 26, 2000	Bluecurve	IT management software	 United States	$37,107,000	[113]
August 1, 2000	Wirespeed Communications	Internet software	 United States	$83,963,000	[114]
August 15, 2000	Hell's Kitchen Systems	Internet software	 United States	$85,624,000	[115]
September 13, 2000	C2Net	Internet software	 United States	$39,983,000	[116]
February 5, 2001	Akopia	Ecommerce software	 United States	—	[117]
February 28, 2001	Planning Technologies	Consulting	 United States	$47,000,000	[118]
February 11, 2002	ArsDigita	Assets and employees	 United States	—	[119]
October 15, 2002	NOCpulse	Software	 United States	—	[120]
December 18, 2003	Sistina Software	GFS, LVM, DM	 United States	$31,000,000	[121]
September 30, 2004	The Netscape Security
-Certain Asts[note 2]	Certain assets	 United States	—	[122]
June 5, 2006	JBoss	Middleware	 France	$420,000,000	[123][124]
June 6, 2007	MetaMatrix	Information management software	 United States	—	[125]
June 19, 2007	Mobicents	Telecommunications software	 United States	—	[126]
March 13, 2008	Amentra	Consulting	 United States	—	[127]
June 4, 2008	Identyx	Software	 United States	—	[128]
September 4, 2008	Qumranet	KVM, RHEV, SPICE	 Israel	$107,000,000	[129]
November 30, 2010	Makara	Enterprise software	 United States	—	[130]
October 4, 2011	Gluster	GlusterFS	 United States	$136,000,000	[131]
June 27, 2012	FuseSource	Enterprise integration software	 United States	—	[132]
August 28, 2012	Polymita	Enterprise software	 Spain	—	[133]
December 20, 2012	ManageIQ	Orchestration software	 United States	$104,000,000	[134]
January 7, 2014	The CentOS Project	CentOS	 United States	—	[135][136]
April 30, 2014	Inktank Storage	Ceph	 United States	$175,000,000	[137]
June 18, 2014	eNovance	OpenStack Integration Services	 France	$95,000,000	[138]
September 18, 2014	FeedHenry	Mobile Application Platform	 Ireland	$82,000,000	[139]
October 16, 2015	Ansible	Configuration management, Orchestration engine	 United States	—	[140]
June 22, 2016	3scale	API management	 United States	—	[141]
May 25, 2017	Codenvy	Cloud software	 United States	—	[142]
July 31, 2017	Permabit	Data deduplication and compression	 United States	—	[143]
January 30, 2018	CoreOS	Management of containerized application:
Container Linux by CoreOS	 United States	$250,000,000	[144]
November 28, 2018	NooBaa	Cloud storage technology	 Israel	—	[145]
January 7, 2021	StackRox	Container management software	 United States	—	[146]
Divestitures
Date	Acquirer	Target company	Target business	Acquirer country	Value (USD)	References
December 14, 1998	Intel Corporation	Red Hat[note 3]	Open-source software	 United States	—	[147]
March 9, 1999	Compaq	Red Hat[note 4]	Open-source software	 United States	—	[148]
March 9, 1999	IBM	Red Hat[note 5]	Open-source software	 United States	—	[149]
March 9, 1999	Novell	Red Hat[note 6]	Open-source software	 United States	—	[150]
 Delix Computer GmbH-Linux Div was acquired from Delix Computer.
 Netscape Security-Certain Asts was acquired from Netscape Security Solutions.
 Intel Corporation acquired a minority stake in Red Hat.
 Compaq acquired a minority stake in Red Hat.
 IBM acquired a minority stake in Red Hat.
 Novell acquired a minority stake in Red Hat"""

DataStax = """DataStax

Article
Talk
Read
Edit
View history

Tools
Appearance hide
Text

Small

Standard

Large
Width

Standard

Wide
Color (beta)

Automatic

Light

Dark
From Wikipedia, the free encyclopedia
DataStax

Logo used since 2023
Company type	Private
Industry	Database Technologies
Genre	Multi-Model DBMS
Founded	April 2010
Founder
Jonathan Ellis
Matt Pfeil
Headquarters	Santa Clara, CA, United States
Key people
Chet Kapoor[1] (CEO)
Davor Bonaci (CTO)
Ed Anuff (CPO)
Don Dixon (CFO)
Brad Gyger (CRO)
Jason McClelland (CMO)
Chris Vogel (CPO)
Number of employees	800+ (June 2022)[2]
Website	www.datastax.com Edit this at Wikidata
DataStax, Inc. is a real-time data for AI company based in Santa Clara, California.[3] Its product Astra DB is a cloud database-as-a-service based on Apache Cassandra. DataStax also offers DataStax Enterprise (DSE), an on-premises database built on Apache Cassandra, and Astra Streaming, a messaging and event streaming cloud service based on Apache Pulsar. As of June 2022, the company has roughly 800 customers distributed in over 50 countries.[4][5][2]

History
DataStax was built on the open source NoSQL database Apache Cassandra. Cassandra was initially developed internally at Facebook to handle large data sets across multiple servers,[6] and was released as an Apache open source project in 2008.[7] In 2010, Jonathan Ellis and Matt Pfeil left Rackspace, where they had worked with Cassandra, to launch Riptano in Austin, Texas.[6][8] Ellis and Pfeil later renamed the company DataStax, and moved its headquarters to Santa Clara, California.[3][9]

The company went on to create its own enterprise version of Cassandra, a NoSQL database called DataStax Enterprise (DSE).[6]

In 2019, Chet Kapoor was named the company's new CEO, taking over from Billy Bosworth.[10]


Original logo
In May 2020, DataStax released Astra DB, a DBaaS for Cassandra applications.[11] In November 2020, DataStax released K8ssandra, an open source distribution of Cassandra on Kubernetes.[12] In December 2020, DataStax released Stargate, an open source data API gateway.[13]

After acquiring streaming event vendor Kesque in January 2021,[14] the company launched Luna Streaming, a data streaming platform for Apache Pulsar.[15] DataStax then rebuilt the Kesque technology into Astra Streaming.[16] The Astra Streaming cloud service became generally available on June 29, 2022.[17] With the release, the company added API-level support for messaging tools Apache Kafka, RabbitMQ and Java Message Service, in addition to Apache Pulsar.[18][19] Astra Streaming can connect to a larger data platform by utilizing DataStax's Astra DB cloud service.[18]

Starting in 2023, DataStax began incorporating artificial intelligence and machine learning into its platform.[20] In January 2023, the company acquired Kaskada, developer of a platform that helps organizations use data for AI applications.[21] DataStax made the formerly proprietary Kaskada technology open source, and integrated it into its Luna ML service, which was launched on May 4, 2023.[22] With the acquisition, former Kaskada CEO Davor Bonaci was named DataStax chief technology officer and executive vice president.[22]

On May 24, 2023, DataStax announced that it would be partnering with ThirdAI to bring large language models to DSE and AstraDB, to help developers develop generative AI applications.[23]

In June 2023, the company announced the development of a GPT-based schema translator in its Astra Streaming cloud service. The Astra Streaming GPT Schema Translator uses generative AI to automatically generate schema mappings, to enable data integration and interoperability between multiple systems and data sources.[24]

On July 18, 2023, the company announced a partnership with Google to make semantic search available in its Astra DB cloud database for developers building generative AI applications.[20]

On September 13, 2023, DataStax launched the LangStream open source project, which works with Astra DB and supports vector databases including Milvus and Pinecone. LangStream enables developers to better work with streaming data sources, using Apache Kafka technology and generative AI to help build event-driven architectures.[25]

In November 2023, DataStax announced RAGStack, a simplified commercial offering for RAG (retrieval-augmented generation) based on LangChain and Astra DB vector search.[26]

On February 25, 2025, IBM announced its intention to acquire DataStax.[27][28]

Products
Astra DB
Astra DB is available on cloud services such as Microsoft Azure, Amazon Web Services, and Google Cloud Platform.[29] In February 2021, DataStax announced the serverless version of Astra DB, offering developers pay-as-you-go data.[30]

In March 2022, DataStax introduced new change data capture (CDC) capabilities to its Astra DB cloud service. Astra DB CDC is powered by Apache Pulsar, which allows developers to manage operational and streaming data in one place.[31] DataStax leads the open-source Starlight, which provides a compatibility layer for different protocols on top of Apache Pulsar.[18]

On February 8, 2023, DataStax launched Astra Block, a cloud-based service based on the Ethereum blockchain to support building Web3 applications, available as part of Astra DB. Astra Block can be used by developers to stream enhanced data from the Ethereum blockchain to build or scale Web3 experiences on Astra DB.[32]

Astra DB supports open source LangChain technology, making it easier for developers to create generative AI applications.[20]

DSE
Version 1.0 of the DataStax Enterprise (DSE), released in October 2011, was the first commercial distribution of the Cassandra database, designed to provide real-time application performance and heavy analytics on the same physical infrastructure.[33][34] It grew to include advanced security controls, graph database models, operational analytics and advanced search capabilities.[35]

In April 2016, the company announced the release of DataStax Enterprise Graph, adding graph data model functionality to DSE.[36]

In March 2017, DataStax announced the release of its DSE platform 5.1, which included improved search capabilities, improved security control, improvements to its Graph data management and improvements to operational analytics performance. DataStax also announced a shift in strategy, with an added focus on customer experience applications. Rather than a new set of technologies, the company started to offer advice on best practice to users of its core DSE platform.[37][35]

In April 2018, DataStax released DSE 6, with the new version focused on businesses using a hybrid cloud computing model, with all the benefits of a distributed cloud database on any public cloud or on-premise, twice the responsiveness and ability to handle twice the throughput.[38][39]

In December 2018, DataStax released DSE 6.7, which offers enterprise customers five key new feature upgrades, including: improved analytics, geospatial search, improved data protection in the cloud, enhanced performance insights and new developer integration tools with Apache Kafka Connector and certified production Docker images.[40]

In April 2020, DataStax released DSE 6.8, offering enterprises new capabilities for bare-metal performance and to support more workloads, and serving as a Kubernetes operator for Cassandra.[41]

DSE 7.0 was introduced in August 2023. It offers enhancements in cloud-native operations and generative AI capabilities, and includes vector search.[42]

Funding and IPO
In September 2014, DataStax raised $106 million in a Series E funding round, raising the total investment in the company to $190 million.[3] On June 15, 2022, the company announced it had raised an additional $115 million, at a $1.6 billion valuation.[2][43]

In 2020, Mergermarket reported that DataStax was preparing for an initial public offering that could launch in 2021.[44] However, in June 2022, DataStax CEO Chet Kapoor said that the company would not rush into an IPO.[2]"""


Apptio = """Apptio, Inc. is a Bellevue, Washington-based company founded in 2007 that develops technology business management (TBM) software as a service (SaaS) applications.[3][4][5] Apptio enterprise apps are designed to assess and communicate the cost of IT services for planning, budgeting and forecasting purposes;[6] Apptio's services offer tools for CIOs to manage technology departments' storage, applications, energy usage, cybersecurity, and reporting obligations;[7] manage the costs of public cloud, migration to public cloud and SaaS portfolios; and adopt and scale Agile across the enterprise.

In 2009, the company was the first investment for Silicon Valley venture capital firm Andreessen Horowitz.[8] The company has approximately 550 customers[9] of various sizes.[10][11] The company went public in September 2016.[12]

On November 11, 2018, it was announced that Apptio would be acquired by the private equity firm Vista Equity Partners for $1.9 billion.[13] That same year, Apptio acquired Digital Fuel, a cloud computing expenditure company, for $42.5 million.[14]

In June 2023, IBM agreed to acquire Apptio from Vista for $4.6 billion.[15] The acquisition was completed on August 10, 2023.[16]

History
The company was founded in 2007 by Sunny Gupta, Kurt Shintaffer and Paul McLachlan.[17][better source needed] Prior to founding the company, Gupta and Shintaffer worked together at iConclude before it was purchased by Opsware in 2007.[8] In November 2007, the company received $7 million in funding from Madrona Venture Group and Greylock Partners while it was operating in stealth mode.[18] Apptio raised an additional $14 million in Series B funding from the newly formed Andreessen Horowitz fund in 2009.[5]

In 2010, the company raised an additional $16.5 million in Series C funding led by Shasta Ventures, with participation from previous investors.[19]

In March 2012, the company raised an additional $50 million in Series D funding to continue scaling, hiring and expanding its services.[20] In 2013, Apptio raised an additional $45 million in Series E funding, bringing its total raised to $136 million from firms including Greylock Partners, Madrona Venture Group, Janus Capital and T. Rowe Price.[3][10]

In March 2014, Apptio opened offices in Sydney and Melbourne, Australia.[21] In 2015, the company announced that it was adding an office in Denver.[22] Apptio added another international office in March 2016, when it announced it would open an office in Paris, France.[23] On September 23, 2016, Apptio raised $96 million with its IPO.[12]

On February 2, 2018, Apptio completed the acquisition of Digital Fuel SV, LLC, a provider of IT business management (ITBM) tools.[9] On November 11, 2018, Apptio entered into definitive agreement to be acquired by Vista Equity Partners for $1.94 billion. Apptio shareholders received $38.00 in cash per share, representing a 53% premium to the unaffected closing price as of November 9, 2018.[24]

In 2019, Apptio acquired Cloudability, a firm that manages cloud spend.[25] Apptio opened its first India office location in August 2019.[26] In May 2020, LeanIX revealed technology collaboration with Apptio to include core IT spending data analytics.[27]

In February 2021, the company acquired Targetprocess, a software platform that specializes in business investments, for agile.[28] In April 2021, Apptio launched a redesigned portfolio of products including: ApptioOne, Targetprocess, and Cloudability.[29] Apptio announced its first certified solution and integration with ServiceNow in July 2021. The following month, Apptio opened its Kraków, Poland office.[30]

In December 2021, the company announced its collaboration with Microsoft to deploy Apptio’s platform on Microsoft Cloud to help enterprises migrate and optimize workloads.[31] Apptio also partnered with IBM to accelerate enterprise transformation and improve hybrid cloud technology that same year.[32]

In June 2023, IBM agreed to acquire Apptio from Vista Equity Partners for $4.6 billion in an all-cash deal.[15] In August, the company announced the completion of the acquisition of Apptio.[16]

Technology Business Management Council
Apptio founded the Technology Business Management Council in 2012,[33] a non-profit organization that had over 2,900 CIOs[34] and other senior IT leaders by 2014.[33] It was reported by the Puget Sound Business Journal that the council "aims to help IT departments run like businesses by setting standards and best practices" .[35] The council held its first conference in November 2013.[11]
"""


documents = [
    RAGDocument("HashiCorp", "https://en.wikipedia.org/wiki/HashiCorp", hashi),
    RAGDocument("DataStax", "https://en.wikipedia.org/wiki/DataStax", DataStax),
    RAGDocument("Apptio", "https://en.wikipedia.org/wiki/Apptio", Apptio),
    RAGDocument("Red Hat", "https://en.wikipedia.org/wiki/Red_Hat", redhat),
]
