from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, '../data/career_data.csv'), sep="\t")
X = df.drop("career", axis=1)
careers = df["career"]

descriptions = {
    "Software Engineer": "Build and maintain software systems and applications",
    "Data Scientist": "Extract insights and build models from complex datasets",
    "Data Analyst": "Analyze data to inform business decisions and strategy",
    "AI Engineer": "Design and deploy artificial intelligence systems",
    "Machine Learning Engineer": "Build and optimize machine learning pipelines",
    "Cybersecurity Analyst": "Protect systems and networks from digital threats",
    "System Administrator": "Manage and maintain IT infrastructure",
    "Network Engineer": "Design and maintain computer networks",
    "Cloud Architect": "Design scalable cloud-based infrastructure",
    "DevOps Engineer": "Bridge development and operations for faster delivery",
    "Game Developer": "Create interactive games and gaming experiences",
    "Mobile App Developer": "Build applications for smartphones and tablets",
    "Web Developer": "Design and build websites and web applications",
    "Embedded Systems Engineer": "Program software for hardware devices",
    "Hardware Engineer": "Design and develop physical computing components",
    "Robotics Engineer": "Build and program automated robotic systems",
    "Telecommunications Engineer": "Design and maintain communication networks",
    "Mechatronics Engineer": "Combine mechanical, electrical and software engineering",
    "Control Systems Engineer": "Design systems that regulate processes automatically",
    "Quantum Computing Researcher": "Research next-generation quantum computing systems",
    "Blockchain Developer": "Build decentralized applications and smart contracts",
    "AR/VR Developer": "Create augmented and virtual reality experiences",
    "Database Administrator": "Manage and optimize databases and data storage",
    "IT Consultant": "Advise organizations on technology strategy",
    "Technical Writer": "Create clear documentation for technical products",
    "Business Analyst": "Bridge business needs and technical solutions",
    "Operations Analyst": "Improve efficiency in business operations",
    "Risk Analyst": "Identify and evaluate financial and operational risks",
    "Strategy Analyst": "Develop long-term plans and competitive strategies",
    "Quantitative Analyst": "Apply mathematics to financial modeling",
    "Actuarial Analyst": "Use statistics to assess financial risk",
    "Supply Chain Analyst": "Optimize the flow of goods and services",
    "Policy Analyst": "Research and evaluate government and organizational policies",
    "Market Research Analyst": "Study market trends and consumer behavior",
    "Compliance Officer": "Ensure organizations follow laws and regulations",
    "Financial Analyst": "Evaluate investment opportunities and financial data",
    "Investment Analyst": "Research and recommend investment strategies",
    "Credit Analyst": "Assess creditworthiness of individuals and businesses",
    "Physicist": "Study the fundamental laws of nature and matter",
    "Chemist": "Research and develop chemical compounds and reactions",
    "Biologist": "Study living organisms and their environments",
    "Microbiologist": "Research microscopic organisms like bacteria and viruses",
    "Geneticist": "Study genes, heredity and genetic variation",
    "Neuroscientist": "Research the brain and nervous system",
    "Epidemiologist": "Track and control disease outbreaks and public health",
    "Biomedical Scientist": "Apply science to medical diagnosis and treatment",
    "Astronomer": "Study celestial objects and the universe",
    "Environmental Scientist": "Study and protect the natural environment",
    "Geologist": "Study the earth's structure and natural processes",
    "Oceanographer": "Research oceans, marine life and underwater systems",
    "Statistician": "Collect and interpret numerical data",
    "Mathematician": "Develop and apply mathematical theories",
    "Research Scientist": "Conduct experiments to advance scientific knowledge",
    "Lab Technician": "Perform tests and experiments in a laboratory",
    "Medical Lab Scientist": "Analyze biological samples for medical diagnosis",
    "Pharmacist": "Dispense medications and advise on their use",
    "Biotechnologist": "Apply biology to develop new products and processes",
    "Aerospace Engineer": "Design aircraft, spacecraft and related systems",
    "Nuclear Engineer": "Work with nuclear energy and radiation technology",
    "Renewable Energy Engineer": "Develop sustainable energy solutions",
    "Civil Engineer": "Design and build infrastructure like roads and bridges",
    "Mechanical Engineer": "Design and build mechanical systems and machines",
    "Electrical Engineer": "Design electrical systems and components",
    "Chemical Engineer": "Apply chemistry to industrial processes",
    "Structural Engineer": "Ensure buildings and structures are safe and stable",
    "Graphic Designer": "Create visual concepts and design materials",
    "UX Designer": "Design intuitive and user-friendly digital experiences",
    "UI Designer": "Create visually appealing interfaces for digital products",
    "Interior Designer": "Design functional and beautiful indoor spaces",
    "Fashion Designer": "Create clothing, accessories and fashion collections",
    "Textile Designer": "Design patterns and materials for fabric and clothing",
    "Ceramic Artist": "Create art and functional objects from clay",
    "Illustrator": "Create drawings and illustrations for various media",
    "Animation Artist": "Bring characters and stories to life through animation",
    "Motion Graphics Designer": "Create animated visual content for media",
    "Set Designer": "Design physical environments for film, TV and theater",
    "Art Director": "Lead the visual style and creative direction of projects",
    "Creative Director": "Oversee the creative vision of campaigns and brands",
    "Exhibition Designer": "Design spaces for museums, galleries and events",
    "Visual Merchandiser": "Create appealing product displays in retail spaces",
    "Photographer": "Capture moments and stories through photography",
    "Videographer": "Record and edit video content for various purposes",
    "Filmmaker": "Write, direct and produce film and video projects",
    "Painter": "Create original artwork using paint and canvas",
    "Sculptor": "Create three-dimensional art from various materials",
    "Writer": "Craft stories, articles, books and other written content",
    "Journalist": "Report, investigate and write news stories",
    "Content Creator": "Produce engaging content for digital platforms",
    "Copywriter": "Write persuasive text for advertising and marketing",
    "Script Writer": "Write scripts for film, TV, podcasts and theater",
    "Podcast Host": "Create and host audio content on various topics",
    "YouTuber": "Create video content for YouTube audiences",
    "Travel Blogger": "Document and share travel experiences online",
    "Food Blogger": "Create content about food, recipes and restaurants",
    "Social Media Manager": "Manage and grow brand presence on social media",
    "Community Manager": "Build and nurture online and offline communities",
    "Brand Manager": "Develop and maintain a company's brand identity",
    "Public Relations Specialist": "Manage public image and media relationships",
    "Influencer": "Build an audience and create sponsored content online",
    "Editor": "Review and improve written or visual content",
    "Entrepreneur": "Build and grow your own business from the ground up",
    "Product Manager": "Lead the development and strategy of a product",
    "Operations Manager": "Oversee day-to-day business operations",
    "Management Consultant": "Help organizations improve performance and strategy",
    "Digital Marketing Specialist": "Drive online growth through digital channels",
    "Marketing Manager": "Plan and execute marketing strategies",
    "Human Resource Manager": "Manage recruitment, culture and employee relations",
    "Business Development Manager": "Find and develop new business opportunities",
    "Project Manager": "Plan, execute and deliver projects on time",
    "Event Manager": "Plan and execute events and experiences",
    "Tourism Manager": "Oversee tourism operations and hospitality services",
    "Real Estate Agent": "Help clients buy, sell and rent properties",
    "Finance Manager": "Oversee financial planning and reporting",
    "Auditor": "Examine financial records for accuracy and compliance",
    "Accountant": "Manage financial records and tax obligations",
    "Doctor": "Diagnose and treat medical conditions and illnesses",
    "Nurse": "Provide patient care and support in healthcare settings",
    "Dentist": "Diagnose and treat dental and oral health issues",
    "Physiotherapist": "Help patients recover physical movement and function",
    "Psychologist": "Study and support mental health and behavior",
    "Psychiatrist": "Diagnose and treat mental health disorders",
    "Nutritionist": "Advise on diet and nutrition for health and wellbeing",
    "Surgeon": "Perform surgical operations to treat medical conditions",
    "Paramedic": "Provide emergency medical care in the field",
    "Occupational Therapist": "Help people regain skills for daily life",
    "Speech Therapist": "Help people with communication and swallowing disorders",
    "Veterinarian": "Provide medical care for animals",
    "Public Health Officer": "Promote and protect community health",
    "Health Data Analyst": "Analyze healthcare data to improve patient outcomes",
    "Teacher": "Educate and inspire students in a school setting",
    "Professor": "Teach and conduct research at university level",
    "Career Counselor": "Guide people in making career and education decisions",
    "School Counselor": "Support students' academic and emotional wellbeing",
    "Social Worker": "Support vulnerable individuals and families",
    "NGO Worker": "Work for nonprofit organizations on social causes",
    "Community Development Officer": "Improve living conditions in local communities",
    "Life Coach": "Help people achieve personal and professional goals",
    "Lawyer": "Provide legal advice and represent clients in court",
    "Judge": "Preside over legal proceedings and deliver verdicts",
    "Legal Consultant": "Advise organizations on legal matters",
    "Policy Maker": "Develop policies that guide organizations and governments",
    "Diplomat": "Represent a country in international relations",
    "Civil Servant": "Work in government administration and public service",
    "Police Officer": "Maintain law and order in communities",
    "Chef": "Create and prepare culinary experiences in a kitchen",
    "Pilot": "Fly and navigate aircraft safely",
    "Athlete": "Compete professionally in sports",
    "Personal Trainer": "Help clients achieve fitness and health goals",
    "Plumber": "Install and repair water and pipe systems",
    "Electrician": "Install and maintain electrical systems",
    "Carpenter": "Build and repair wooden structures and furniture",
    "Architect": "Design buildings that are safe, functional and beautiful",
    "Urban Planner": "Plan the development of cities and communities",
    "Landscape Designer": "Design outdoor spaces and gardens",
    "Musician": "Create and perform music professionally",
    "Music Producer": "Record, mix and produce musical tracks",
    "Sound Engineer": "Manage audio recording and production",
    "Actor": "Perform characters in film, TV and theater",
    "Dance Choreographer": "Create and direct dance performances",
    "Theater Director": "Lead the creative vision of stage productions",
    "Travel Guide": "Lead and educate tourists about destinations",
    "Home Maker": "Manage a household and care for family",
    "Florist": "Design and arrange flowers for events and retail",
    "Librarian": "Manage library collections and help people find information",
    "Archivist": "Preserve and manage historical records and documents",
    "Translator": "Convert written content between languages",
    "Interpreter": "Translate spoken language in real time",
}

@app.route("/", methods=["GET", "POST"])
def index():
    top5 = None
    if request.method == "POST":

        # Get user inputs from form
        user_input = {}
        for col in X.columns:
            user_input[col] = [int(request.form.get(col, 1))]

        new_input = pd.DataFrame(user_input, columns=X.columns)

        # Compute cosine similarity
        similarity_scores = cosine_similarity(new_input, X)[0]

        # Use a copy to avoid modifying the global dataframe
        results = df.copy()
        results["similarity"] = similarity_scores

        # Get Top 5 careers
        top5 = results.sort_values(
            by="similarity", ascending=False
        ).head(5)[["career", "similarity"]].values.tolist()

    return render_template("index.html", top5=top5, descriptions=descriptions)

if __name__ == "__main__":
    app.run(debug=True)