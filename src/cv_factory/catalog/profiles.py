"""Built-in career profile catalog."""

from __future__ import annotations

from cv_factory.models.catalog import (
    CareerProfile,
    EducationOption,
    JobTitleOption,
    Seniority,
    Skill,
    SkillCategory,
)


CAREER_PROFILES: tuple[CareerProfile, ...] = (
    CareerProfile(
        key="backend_developer",
        label="Backend Developer",
        family="software",
        titles=[
            JobTitleOption(
                title="Junior Backend Developer",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Backend Developer",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Backend Engineer",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Lead Backend Engineer",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Computer Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Software Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Higher Technician in Web Application Development",
                institution_type="vocational_training",
                level="vocational",
                relevance=4,
            ),
            EducationOption(
                title="Backend Development Bootcamp",
                institution_type="bootcamp",
                level="bootcamp",
                relevance=3,
            ),
        ],
        skills=[
            Skill(name="Python", category=SkillCategory.CORE, weight=5),
            Skill(name="FastAPI", category=SkillCategory.CORE, weight=5),
            Skill(name="REST APIs", category=SkillCategory.CORE, weight=5),
            Skill(name="PostgreSQL", category=SkillCategory.CORE, weight=4),
            Skill(name="Docker", category=SkillCategory.CORE, weight=4),
            Skill(name="Git", category=SkillCategory.TOOL, weight=5),
            Skill(name="Redis", category=SkillCategory.SECONDARY, weight=3),
            Skill(name="CI/CD", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Unit Testing", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Problem Solving", category=SkillCategory.SOFT, weight=4),
        ],
        possible_industries=[
            "SaaS",
            "FinTech",
            "HealthTech",
            "E-commerce",
            "Enterprise Software",
        ],
    ),
    CareerProfile(
        key="frontend_developer",
        label="Frontend Developer",
        family="software",
        titles=[
            JobTitleOption(
                title="Junior Frontend Developer",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Frontend Developer",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Frontend Engineer",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Lead Frontend Engineer",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Computer Science",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Software Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Higher Technician in Web Application Development",
                institution_type="vocational_training",
                level="vocational",
                relevance=4,
            ),
            EducationOption(
                title="Frontend Development Bootcamp",
                institution_type="bootcamp",
                level="bootcamp",
                relevance=3,
            ),
        ],
        skills=[
            Skill(name="JavaScript", category=SkillCategory.CORE, weight=5),
            Skill(name="TypeScript", category=SkillCategory.CORE, weight=5),
            Skill(name="React", category=SkillCategory.CORE, weight=5),
            Skill(name="HTML", category=SkillCategory.CORE, weight=5),
            Skill(name="CSS", category=SkillCategory.CORE, weight=5),
            Skill(name="Responsive Design", category=SkillCategory.CORE, weight=4),
            Skill(name="REST APIs", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Git", category=SkillCategory.TOOL, weight=5),
            Skill(name="Testing Library", category=SkillCategory.SECONDARY, weight=3),
            Skill(name="Communication", category=SkillCategory.SOFT, weight=4),
        ],
        possible_industries=["SaaS", "E-commerce", "Media", "EdTech", "TravelTech"],
    ),
    CareerProfile(
        key="fullstack_developer",
        label="Full Stack Developer",
        family="software",
        titles=[
            JobTitleOption(
                title="Junior Full Stack Developer",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Full Stack Developer",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Full Stack Engineer",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Technical Lead", seniority=Seniority.LEAD, min_years_experience=8
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Computer Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Software Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Higher Technician in Web Application Development",
                institution_type="vocational_training",
                level="vocational",
                relevance=4,
            ),
            EducationOption(
                title="Full Stack Web Development Bootcamp",
                institution_type="bootcamp",
                level="bootcamp",
                relevance=3,
            ),
        ],
        skills=[
            Skill(name="TypeScript", category=SkillCategory.CORE, weight=5),
            Skill(name="React", category=SkillCategory.CORE, weight=5),
            Skill(name="Node.js", category=SkillCategory.CORE, weight=5),
            Skill(name="REST APIs", category=SkillCategory.CORE, weight=5),
            Skill(name="PostgreSQL", category=SkillCategory.CORE, weight=4),
            Skill(name="Docker", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Git", category=SkillCategory.TOOL, weight=5),
            Skill(name="CI/CD", category=SkillCategory.SECONDARY, weight=3),
            Skill(name="System Design", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Adaptability", category=SkillCategory.SOFT, weight=4),
        ],
        possible_industries=[
            "SaaS",
            "Startups",
            "E-commerce",
            "Marketplace Platforms",
            "Internal Tools",
        ],
    ),
    CareerProfile(
        key="data_analyst",
        label="Data Analyst",
        family="data",
        titles=[
            JobTitleOption(
                title="Junior Data Analyst",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Data Analyst", seniority=Seniority.MID, min_years_experience=2
            ),
            JobTitleOption(
                title="Senior Data Analyst",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Analytics Lead", seniority=Seniority.LEAD, min_years_experience=8
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Statistics",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Economics",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Bachelor's Degree in Business Analytics",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Master's Degree in Data Analytics",
                institution_type="university",
                level="master",
                relevance=5,
            ),
        ],
        skills=[
            Skill(name="SQL", category=SkillCategory.CORE, weight=5),
            Skill(name="Excel", category=SkillCategory.CORE, weight=5),
            Skill(name="Python", category=SkillCategory.CORE, weight=4),
            Skill(name="Pandas", category=SkillCategory.CORE, weight=4),
            Skill(name="Data Visualization", category=SkillCategory.CORE, weight=5),
            Skill(name="Power BI", category=SkillCategory.TOOL, weight=4),
            Skill(name="Tableau", category=SkillCategory.TOOL, weight=4),
            Skill(name="Statistics", category=SkillCategory.CORE, weight=5),
            Skill(
                name="Business Reporting", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(name="Analytical Thinking", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=["Retail", "Banking", "Marketing", "SaaS", "Healthcare"],
    ),
    CareerProfile(
        key="data_scientist",
        label="Data Scientist",
        family="data",
        titles=[
            JobTitleOption(
                title="Junior Data Scientist",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Data Scientist", seniority=Seniority.MID, min_years_experience=2
            ),
            JobTitleOption(
                title="Senior Data Scientist",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Lead Data Scientist",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Mathematics",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Computer Science",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Master's Degree in Data Science",
                institution_type="university",
                level="master",
                relevance=5,
            ),
            EducationOption(
                title="Master's Degree in Artificial Intelligence",
                institution_type="university",
                level="master",
                relevance=5,
            ),
        ],
        skills=[
            Skill(name="Python", category=SkillCategory.CORE, weight=5),
            Skill(name="Machine Learning", category=SkillCategory.CORE, weight=5),
            Skill(name="Statistics", category=SkillCategory.CORE, weight=5),
            Skill(name="Pandas", category=SkillCategory.CORE, weight=4),
            Skill(name="scikit-learn", category=SkillCategory.CORE, weight=4),
            Skill(name="SQL", category=SkillCategory.CORE, weight=4),
            Skill(
                name="Data Visualization", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(name="Experimentation", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Model Evaluation", category=SkillCategory.CORE, weight=5),
            Skill(name="Critical Thinking", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=["FinTech", "HealthTech", "Retail", "Insurance", "SaaS"],
    ),
    CareerProfile(
        key="machine_learning_engineer",
        label="Machine Learning Engineer",
        family="data",
        titles=[
            JobTitleOption(
                title="Junior Machine Learning Engineer",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Machine Learning Engineer",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Machine Learning Engineer",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="ML Platform Lead",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Computer Science",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Computer Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Master's Degree in Artificial Intelligence",
                institution_type="university",
                level="master",
                relevance=5,
            ),
            EducationOption(
                title="Master's Degree in Machine Learning",
                institution_type="university",
                level="master",
                relevance=5,
            ),
        ],
        skills=[
            Skill(name="Python", category=SkillCategory.CORE, weight=5),
            Skill(name="PyTorch", category=SkillCategory.CORE, weight=5),
            Skill(name="TensorFlow", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="MLOps", category=SkillCategory.CORE, weight=5),
            Skill(name="Docker", category=SkillCategory.CORE, weight=4),
            Skill(name="Kubernetes", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Model Deployment", category=SkillCategory.CORE, weight=5),
            Skill(name="Feature Engineering", category=SkillCategory.CORE, weight=4),
            Skill(
                name="Experiment Tracking", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(name="Problem Solving", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=[
            "AI Platforms",
            "FinTech",
            "HealthTech",
            "Cybersecurity",
            "E-commerce",
        ],
    ),
    CareerProfile(
        key="devops_engineer",
        label="DevOps Engineer",
        family="infrastructure",
        titles=[
            JobTitleOption(
                title="Junior DevOps Engineer",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="DevOps Engineer", seniority=Seniority.MID, min_years_experience=2
            ),
            JobTitleOption(
                title="Senior DevOps Engineer",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Platform Engineering Lead",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Computer Engineering",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Information Systems",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Higher Technician in Network Systems Administration",
                institution_type="vocational_training",
                level="vocational",
                relevance=5,
            ),
            EducationOption(
                title="Cloud Engineering Certification",
                institution_type="professional_certification",
                level="certificate",
                relevance=4,
            ),
        ],
        skills=[
            Skill(name="Linux", category=SkillCategory.CORE, weight=5),
            Skill(name="Docker", category=SkillCategory.CORE, weight=5),
            Skill(name="Kubernetes", category=SkillCategory.CORE, weight=5),
            Skill(name="Terraform", category=SkillCategory.CORE, weight=5),
            Skill(name="CI/CD", category=SkillCategory.CORE, weight=5),
            Skill(name="AWS", category=SkillCategory.CORE, weight=4),
            Skill(name="Monitoring", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Scripting", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Networking", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Incident Response", category=SkillCategory.SOFT, weight=4),
        ],
        possible_industries=[
            "Cloud Services",
            "SaaS",
            "Banking",
            "Telecom",
            "Cybersecurity",
        ],
    ),
    CareerProfile(
        key="cybersecurity_analyst",
        label="Cybersecurity Analyst",
        family="security",
        titles=[
            JobTitleOption(
                title="Junior Cybersecurity Analyst",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Cybersecurity Analyst",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Security Analyst",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Security Operations Lead",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Cybersecurity",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Computer Engineering",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Higher Technician in Network Systems Administration",
                institution_type="vocational_training",
                level="vocational",
                relevance=4,
            ),
            EducationOption(
                title="Security Operations Certification",
                institution_type="professional_certification",
                level="certificate",
                relevance=4,
            ),
        ],
        skills=[
            Skill(name="Network Security", category=SkillCategory.CORE, weight=5),
            Skill(name="SIEM", category=SkillCategory.CORE, weight=5),
            Skill(name="Incident Response", category=SkillCategory.CORE, weight=5),
            Skill(
                name="Vulnerability Assessment", category=SkillCategory.CORE, weight=4
            ),
            Skill(name="Linux", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Python", category=SkillCategory.SECONDARY, weight=3),
            Skill(name="Threat Analysis", category=SkillCategory.CORE, weight=5),
            Skill(
                name="Firewall Management", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(
                name="Security Reporting", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(name="Attention to Detail", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=[
            "Banking",
            "Telecom",
            "Government",
            "Consulting",
            "Cloud Services",
        ],
    ),
    CareerProfile(
        key="product_manager",
        label="Product Manager",
        family="product",
        titles=[
            JobTitleOption(
                title="Associate Product Manager",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Product Manager", seniority=Seniority.MID, min_years_experience=2
            ),
            JobTitleOption(
                title="Senior Product Manager",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Product Lead", seniority=Seniority.LEAD, min_years_experience=8
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Business Administration",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Bachelor's Degree in Computer Science",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Master's Degree in Product Management",
                institution_type="university",
                level="master",
                relevance=5,
            ),
            EducationOption(
                title="MBA",
                institution_type="university",
                level="master",
                relevance=4,
            ),
        ],
        skills=[
            Skill(name="Product Strategy", category=SkillCategory.CORE, weight=5),
            Skill(name="Roadmapping", category=SkillCategory.CORE, weight=5),
            Skill(name="User Research", category=SkillCategory.CORE, weight=4),
            Skill(name="Agile Methodologies", category=SkillCategory.CORE, weight=4),
            Skill(name="Data Analysis", category=SkillCategory.CORE, weight=4),
            Skill(name="Stakeholder Management", category=SkillCategory.SOFT, weight=5),
            Skill(name="Prioritization", category=SkillCategory.CORE, weight=5),
            Skill(name="A/B Testing", category=SkillCategory.SECONDARY, weight=3),
            Skill(name="Market Research", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Communication", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=[
            "SaaS",
            "FinTech",
            "Marketplace Platforms",
            "E-commerce",
            "Mobile Apps",
        ],
    ),
    CareerProfile(
        key="ux_ui_designer",
        label="UX/UI Designer",
        family="design",
        titles=[
            JobTitleOption(
                title="Junior UX/UI Designer",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="UX/UI Designer", seniority=Seniority.MID, min_years_experience=2
            ),
            JobTitleOption(
                title="Senior Product Designer",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Design Lead", seniority=Seniority.LEAD, min_years_experience=8
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Design",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Human-Computer Interaction",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="UX/UI Design Bootcamp",
                institution_type="bootcamp",
                level="bootcamp",
                relevance=4,
            ),
            EducationOption(
                title="Master's Degree in Interaction Design",
                institution_type="university",
                level="master",
                relevance=5,
            ),
        ],
        skills=[
            Skill(name="Figma", category=SkillCategory.TOOL, weight=5),
            Skill(name="User Research", category=SkillCategory.CORE, weight=5),
            Skill(name="Wireframing", category=SkillCategory.CORE, weight=5),
            Skill(name="Prototyping", category=SkillCategory.CORE, weight=5),
            Skill(name="Design Systems", category=SkillCategory.CORE, weight=4),
            Skill(name="Usability Testing", category=SkillCategory.CORE, weight=4),
            Skill(
                name="Information Architecture",
                category=SkillCategory.SECONDARY,
                weight=4,
            ),
            Skill(name="Accessibility", category=SkillCategory.SECONDARY, weight=4),
            Skill(name="Visual Design", category=SkillCategory.CORE, weight=5),
            Skill(name="Empathy", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=[
            "SaaS",
            "Mobile Apps",
            "E-commerce",
            "Banking",
            "HealthTech",
        ],
    ),
    CareerProfile(
        key="digital_marketing_specialist",
        label="Digital Marketing Specialist",
        family="marketing",
        titles=[
            JobTitleOption(
                title="Junior Digital Marketing Specialist",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Digital Marketing Specialist",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Digital Marketing Manager",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
            JobTitleOption(
                title="Growth Marketing Lead",
                seniority=Seniority.LEAD,
                min_years_experience=8,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Marketing",
                institution_type="university",
                level="bachelor",
                relevance=5,
            ),
            EducationOption(
                title="Bachelor's Degree in Business Administration",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Master's Degree in Digital Marketing",
                institution_type="university",
                level="master",
                relevance=5,
            ),
            EducationOption(
                title="Professional Certification in SEO and SEM",
                institution_type="professional_certification",
                level="certificate",
                relevance=4,
            ),
        ],
        skills=[
            Skill(name="SEO", category=SkillCategory.CORE, weight=5),
            Skill(name="SEM", category=SkillCategory.CORE, weight=5),
            Skill(name="Google Ads", category=SkillCategory.TOOL, weight=5),
            Skill(name="Google Analytics", category=SkillCategory.TOOL, weight=5),
            Skill(name="Content Marketing", category=SkillCategory.CORE, weight=4),
            Skill(name="Email Marketing", category=SkillCategory.SECONDARY, weight=4),
            Skill(
                name="Conversion Rate Optimization",
                category=SkillCategory.CORE,
                weight=4,
            ),
            Skill(
                name="Marketing Automation", category=SkillCategory.SECONDARY, weight=3
            ),
            Skill(
                name="Campaign Reporting", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(name="Creativity", category=SkillCategory.SOFT, weight=4),
        ],
        possible_industries=["E-commerce", "Agencies", "SaaS", "Retail", "Education"],
    ),
    CareerProfile(
        key="sales_development_representative",
        label="Sales Development Representative",
        family="sales",
        titles=[
            JobTitleOption(
                title="Sales Development Representative",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Business Development Representative",
                seniority=Seniority.JUNIOR,
                min_years_experience=1,
            ),
            JobTitleOption(
                title="Account Executive",
                seniority=Seniority.MID,
                min_years_experience=2,
            ),
            JobTitleOption(
                title="Senior Account Executive",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Business Administration",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Bachelor's Degree in Marketing",
                institution_type="university",
                level="bachelor",
                relevance=4,
            ),
            EducationOption(
                title="Professional Certification in Sales Development",
                institution_type="professional_certification",
                level="certificate",
                relevance=3,
            ),
            EducationOption(
                title="Higher Technician in Sales and Commercial Management",
                institution_type="vocational_training",
                level="vocational",
                relevance=4,
            ),
        ],
        skills=[
            Skill(name="Prospecting", category=SkillCategory.CORE, weight=5),
            Skill(name="Cold Outreach", category=SkillCategory.CORE, weight=5),
            Skill(name="CRM Management", category=SkillCategory.CORE, weight=5),
            Skill(name="Lead Qualification", category=SkillCategory.CORE, weight=5),
            Skill(name="Salesforce", category=SkillCategory.TOOL, weight=4),
            Skill(name="HubSpot", category=SkillCategory.TOOL, weight=4),
            Skill(
                name="Pipeline Management", category=SkillCategory.SECONDARY, weight=4
            ),
            Skill(name="Negotiation", category=SkillCategory.SOFT, weight=4),
            Skill(name="Communication", category=SkillCategory.SOFT, weight=5),
            Skill(name="Resilience", category=SkillCategory.SOFT, weight=5),
        ],
        possible_industries=[
            "SaaS",
            "B2B Services",
            "Consulting",
            "FinTech",
            "HR Tech",
        ],
    ),
)


__all__ = ["CAREER_PROFILES"]
