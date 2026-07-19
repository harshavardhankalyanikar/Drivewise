"""
Generate sample car brochure PDFs for DriveWise.

No real brochure files were provided with the project brief, and the referenced
Kaggle dataset page only contains the problem statement (no downloadable files).
This script produces realistic, structured brochure PDFs -- each with proper
headings, sections, and a specifications table -- so the ingestion, chunking,
embedding, retrieval, and re-ranking pipeline can be exercised end-to-end with
real PDF parsing (not placeholder text).

Run:
    python scripts/generate_sample_brochures.py
"""

from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "brochures")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(
        ParagraphStyle(
            name="BrochureTitle",
            parent=ss["Title"],
            fontSize=22,
            textColor=colors.HexColor("#0B3D91"),
        )
    )
    ss.add(
        ParagraphStyle(
            name="Section",
            parent=ss["Heading1"],
            fontSize=15,
            spaceBefore=18,
            spaceAfter=8,
            textColor=colors.HexColor("#0B3D91"),
        )
    )
    ss.add(
        ParagraphStyle(
            name="SubSection",
            parent=ss["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1B5E20"),
        )
    )
    ss.add(
        ParagraphStyle(
            name="Body",
            parent=ss["BodyText"],
            fontSize=10.5,
            leading=15,
        )
    )
    return ss


def _spec_table(rows, styles):
    data = [["Specification", "Value"]] + rows
    t = Table(data, colWidths=[7 * cm, 8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build_brochure(path, brand, model, variants, sections):
    """sections: list of (section_title, [ (subheading, paragraph_text, table_rows_or_None) ]) """
    styles = _styles()
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        title=f"{brand} {model} Brochure",
    )
    story = []
    story.append(Paragraph(f"{brand} {model}", styles["BrochureTitle"]))
    story.append(Paragraph("Official Product Brochure", styles["Body"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Available Variants: " + ", ".join(variants), styles["Body"]))
    story.append(PageBreak())

    for section_title, subsections in sections:
        story.append(Paragraph(section_title, styles["Section"]))
        for subheading, paragraph, table_rows in subsections:
            if subheading:
                story.append(Paragraph(subheading, styles["SubSection"]))
            if paragraph:
                story.append(Paragraph(paragraph, styles["Body"]))
                story.append(Spacer(1, 6))
            if table_rows:
                story.append(_spec_table(table_rows, styles))
                story.append(Spacer(1, 10))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Wrote {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------- Hyundai Creta
    build_brochure(
        os.path.join(OUTPUT_DIR, "hyundai_creta_2026.pdf"),
        "Hyundai", "Creta",
        variants=["E", "EX", "S", "SX", "SX(O)"],
        sections=[
            ("Engine and Performance", [
                (
                    "Petrol and Diesel Options",
                    "The Creta is offered with a 1.5L MPi petrol engine and a 1.5L CRDi diesel "
                    "engine. A 1.5L turbo-petrol variant is available exclusively on the SX(O) trim, "
                    "paired with a 7-speed dual-clutch automatic transmission.",
                    [
                        ["Engine Displacement (Petrol)", "1497 cc"],
                        ["Engine Displacement (Diesel)", "1493 cc"],
                        ["Engine Displacement (Turbo Petrol)", "1353 cc"],
                        ["Max Power (Turbo Petrol)", "160 PS @ 5500 rpm"],
                        ["Max Torque (Turbo Petrol)", "253 Nm @ 1500-3200 rpm"],
                        ["Transmission Options", "6-Speed Manual, CVT, 7-Speed DCT, 6-Speed Torque Converter Automatic"],
                        ["Drivetrain", "Front Wheel Drive (FWD)"],
                    ],
                ),
            ]),
            ("Mileage and Fuel Efficiency", [
                (
                    "ARAI Certified Mileage",
                    "The Creta diesel variant with manual transmission delivers the best-in-class "
                    "fuel efficiency, making it ideal for long highway drives.",
                    [
                        ["Mileage - Petrol MT", "17.4 km/l"],
                        ["Mileage - Petrol CVT", "16.8 km/l"],
                        ["Mileage - Diesel MT", "21.0 km/l"],
                        ["Mileage - Diesel AT", "18.4 km/l"],
                        ["Mileage - Turbo Petrol DCT", "16.2 km/l"],
                        ["Fuel Tank Capacity", "50 litres"],
                    ],
                ),
            ]),
            ("Safety Features", [
                (
                    "ADAS Level 2 and Passive Safety",
                    "The SX and SX(O) variants come equipped with Hyundai SmartSense, offering "
                    "Level 2 Advanced Driver Assistance Systems (ADAS) including Forward Collision "
                    "Avoidance Assist, Lane Keeping Assist, Smart Cruise Control, and Blind Spot "
                    "Collision Avoidance Assist. All variants get a minimum of 6 airbags as standard.",
                    [
                        ["ADAS", "Available on SX and SX(O) (Level 2)"],
                        ["Airbags (Standard across range)", "6 (Dual Front, Side, Curtain)"],
                        ["ESC (Electronic Stability Control)", "Standard on all variants"],
                        ["ABS with EBD", "Standard on all variants"],
                        ["Hill Assist Control", "Standard on S, SX, SX(O)"],
                        ["360-Degree Camera", "Available on SX(O)"],
                        ["Tyre Pressure Monitoring System", "Standard on all variants"],
                        ["Global NCAP Rating", "5-Star (Adult Occupant Protection)"],
                    ],
                ),
            ]),
            ("Dimensions", [
                (
                    None,
                    "The Creta's dimensions strike a balance between a compact footprint for city "
                    "driving and a spacious cabin for family use.",
                    [
                        ["Length", "4330 mm"],
                        ["Width", "1790 mm"],
                        ["Height", "1635 mm"],
                        ["Wheelbase", "2610 mm"],
                        ["Ground Clearance", "190 mm"],
                        ["Boot Space", "433 litres"],
                        ["Seating Capacity", "5"],
                    ],
                ),
            ]),
            ("Interior and Comfort", [
                (
                    "Sunroof and Cabin Features",
                    "A single-pane electric sunroof is standard from the S variant upwards, while "
                    "ventilated front seats and a wireless charging pad are offered from the SX "
                    "variant onward. Dual-zone climate control is exclusive to the SX(O) variant.",
                    [
                        ["Sunroof", "Standard from S variant and above"],
                        ["Ventilated Front Seats", "Available from SX variant"],
                        ["Climate Control", "Dual-zone (SX(O) only), Single-zone (others)"],
                        ["Wireless Phone Charger", "Available from SX variant"],
                        ["Upholstery", "Fabric (E, EX, S), Leatherette (SX, SX(O))"],
                    ],
                ),
            ]),
            ("Infotainment and Connectivity", [
                (
                    None,
                    "A 10.25-inch touchscreen infotainment system with Bose premium sound and "
                    "wireless Android Auto and Apple CarPlay comes on the SX(O) trim.",
                    [
                        ["Touchscreen Size", "10.25 inches (SX, SX(O)); 8 inches (S)"],
                        ["Sound System", "Bose 8-speaker (SX(O) only)"],
                        ["Android Auto / Apple CarPlay", "Wireless, from S variant upward"],
                        ["Digital Instrument Cluster", "10.25-inch fully digital (SX, SX(O))"],
                        ["Connected Car Tech", "Hyundai Bluelink, standard from S variant"],
                    ],
                ),
            ]),
        ],
    )

    # ---------------------------------------------------------------- Tata Nexon
    build_brochure(
        os.path.join(OUTPUT_DIR, "tata_nexon_2026.pdf"),
        "Tata", "Nexon",
        variants=["Smart", "Pure", "Creative", "Fearless", "Fearless+"],
        sections=[
            ("Engine and Performance", [
                (
                    "Petrol, Turbo-Petrol and Diesel",
                    "The Nexon is available with a 1.2L turbo-petrol engine and a 1.5L turbo-diesel "
                    "engine, both mated to 6-speed manual or automatic transmissions (AMT/DCT/TCU "
                    "depending on the fuel type).",
                    [
                        ["Engine Displacement (Turbo Petrol)", "1199 cc"],
                        ["Engine Displacement (Turbo Diesel)", "1497 cc"],
                        ["Max Power (Petrol)", "120 PS @ 5500 rpm"],
                        ["Max Power (Diesel)", "115 PS @ 3750 rpm"],
                        ["Transmission Options", "6-Speed Manual, 6-Speed AMT, 7-Speed DCT (Petrol)"],
                        ["Drivetrain", "Front Wheel Drive (FWD)"],
                    ],
                ),
            ]),
            ("Mileage and Fuel Efficiency", [
                (
                    None,
                    "The diesel manual variant remains the most fuel-efficient option in the Nexon lineup.",
                    [
                        ["Mileage - Petrol MT", "17.05 km/l"],
                        ["Mileage - Petrol DCT", "17.66 km/l"],
                        ["Mileage - Diesel MT", "24.08 km/l"],
                        ["Mileage - Diesel AMT", "23.6 km/l"],
                        ["Fuel Tank Capacity", "44 litres"],
                    ],
                ),
            ]),
            ("Safety Features", [
                (
                    "5-Star Global NCAP and ADAS",
                    "The Nexon was among the first Indian-made cars to achieve a 5-star Global NCAP "
                    "safety rating. The Fearless+ variant offers Level 2 ADAS with adaptive cruise "
                    "control, forward collision warning, and automatic emergency braking.",
                    [
                        ["Global NCAP Rating", "5-Star (Adult and Child Occupant Protection)"],
                        ["ADAS", "Available on Fearless+ (Level 2)"],
                        ["Airbags (Standard across range)", "6 (Dual Front, Side, Curtain)"],
                        ["ESC", "Standard on all variants"],
                        ["Disc Brakes (All 4 wheels)", "Available on Fearless and Fearless+"],
                        ["360-Degree Camera", "Available on Fearless+"],
                    ],
                ),
            ]),
            ("Dimensions", [
                (
                    None,
                    "The Nexon offers a sub-4-metre footprint, qualifying it for a compact SUV tax "
                    "bracket, while retaining a tall stance and generous boot space.",
                    [
                        ["Length", "3993 mm"],
                        ["Width", "1811 mm"],
                        ["Height", "1616 mm (1620 mm with roof rails)"],
                        ["Wheelbase", "2498 mm"],
                        ["Ground Clearance", "209 mm"],
                        ["Boot Space", "382 litres (350 litres for EV variant)"],
                        ["Seating Capacity", "5"],
                    ],
                ),
            ]),
            ("Interior and Comfort", [
                (
                    "Sunroof and Comfort Features",
                    "An electric sunroof is available from the Creative variant onward. Ventilated "
                    "seats are offered as an accessory on the Fearless+ trim.",
                    [
                        ["Sunroof", "Available from Creative variant and above"],
                        ["Ventilated Seats", "Available on Fearless+ (as an accessory pack)"],
                        ["Climate Control", "Automatic single-zone from Creative variant"],
                        ["Wireless Phone Charger", "Available from Fearless variant"],
                        ["Upholstery", "Fabric (Smart, Pure); Premium Fabric/Leatherette (Creative, Fearless, Fearless+)"],
                    ],
                ),
            ]),
            ("Infotainment and Connectivity", [
                (
                    None,
                    "The Nexon features a 10.25-inch floating touchscreen with a JBL sound system "
                    "on the top-spec Fearless+ variant, along with an Arcade.ai voice assistant.",
                    [
                        ["Touchscreen Size", "10.25 inches (Fearless, Fearless+)"],
                        ["Sound System", "JBL 9-speaker (Fearless+ only)"],
                        ["Android Auto / Apple CarPlay", "Wireless, from Creative variant"],
                        ["Digital Instrument Cluster", "10.25-inch fully digital (Fearless+)"],
                        ["Connected Car Tech", "Tata iRA Connected Car Tech, standard from Pure variant"],
                    ],
                ),
            ]),
        ],
    )

    # ---------------------------------------------------------------- Maruti Suzuki Baleno
    build_brochure(
        os.path.join(OUTPUT_DIR, "maruti_baleno_2026.pdf"),
        "Maruti Suzuki", "Baleno",
        variants=["Sigma", "Delta", "Zeta", "Alpha"],
        sections=[
            ("Engine and Performance", [
                (
                    None,
                    "The Baleno is powered exclusively by a 1.2L naturally aspirated K-series petrol "
                    "engine, offered with a 5-speed manual or a 5-speed AMT (CVT on the Alpha variant).",
                    [
                        ["Engine Displacement", "1197 cc"],
                        ["Max Power", "90 PS @ 6000 rpm"],
                        ["Max Torque", "113 Nm @ 4400 rpm"],
                        ["Transmission Options", "5-Speed Manual, 5-Speed AMT, CVT (Alpha only)"],
                        ["Drivetrain", "Front Wheel Drive (FWD)"],
                    ],
                ),
            ]),
            ("Mileage and Fuel Efficiency", [
                (
                    None,
                    "The Baleno CNG variant, available on Sigma and Delta trims, offers the best "
                    "running cost per kilometre in the segment.",
                    [
                        ["Mileage - Petrol MT", "22.35 km/l"],
                        ["Mileage - Petrol AMT", "22.94 km/l"],
                        ["Mileage - CNG", "30.61 km/kg"],
                        ["Fuel Tank Capacity", "37 litres"],
                    ],
                ),
            ]),
            ("Safety Features", [
                (
                    None,
                    "The Baleno does not offer ADAS on any variant. All variants get dual airbags "
                    "as standard, with 6 airbags offered from the Zeta variant onward.",
                    [
                        ["ADAS", "Not available on any variant"],
                        ["Airbags (Sigma, Delta)", "2 (Dual Front)"],
                        ["Airbags (Zeta, Alpha)", "6 (Dual Front, Side, Curtain)"],
                        ["ESC", "Standard from Delta variant upward"],
                        ["Global NCAP Rating", "3-Star (Adult Occupant Protection)"],
                    ],
                ),
            ]),
            ("Dimensions", [
                (
                    None,
                    "As a premium hatchback, the Baleno prioritises cabin space and boot capacity "
                    "over a tall SUV-like stance.",
                    [
                        ["Length", "3990 mm"],
                        ["Width", "1745 mm"],
                        ["Height", "1500 mm"],
                        ["Wheelbase", "2520 mm"],
                        ["Ground Clearance", "170 mm"],
                        ["Boot Space", "318 litres"],
                        ["Seating Capacity", "5"],
                    ],
                ),
            ]),
            ("Interior and Comfort", [
                (
                    None,
                    "The Baleno does not offer a sunroof on any variant. Automatic climate control "
                    "is standard from the Zeta variant onward.",
                    [
                        ["Sunroof", "Not available on any variant"],
                        ["Climate Control", "Automatic from Zeta variant, manual on Sigma/Delta"],
                        ["Wireless Phone Charger", "Available on Alpha variant only"],
                        ["Upholstery", "Fabric (all variants)"],
                    ],
                ),
            ]),
            ("Infotainment and Connectivity", [
                (
                    None,
                    "A 9-inch SmartPlay Pro+ touchscreen with Arkamys sound tuning is offered from "
                    "the Zeta variant onward, with wireless Android Auto and Apple CarPlay.",
                    [
                        ["Touchscreen Size", "9 inches (Zeta, Alpha); 4.2-inch MID (Sigma, Delta)"],
                        ["Sound System", "Arkamys-tuned 4-speaker (Zeta, Alpha)"],
                        ["Android Auto / Apple CarPlay", "Wireless, from Zeta variant"],
                        ["Connected Car Tech", "Suzuki Connect, standard from Delta variant"],
                    ],
                ),
            ]),
        ],
    )


if __name__ == "__main__":
    main()
