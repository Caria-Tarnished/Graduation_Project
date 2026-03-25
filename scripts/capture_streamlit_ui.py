# -*- coding: utf-8 -*-
import asyncio
import os
from playwright.async_api import async_playwright

async def capture_screenshots():
    BASE_DIR = "e:/Projects/Graduation_Project"
    OUTPUT_DIR = os.path.join(BASE_DIR, "thesis_assets", "screenshots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting browser...")
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        # Set viewport to a good resolution for thesis screenshots
        context = await browser.new_context(viewport={'width': 1440, 'height': 900})
        page = await context.new_page()
        
        # 1. Chat Interface (Home)
        print("Navigating to Chat Interface...")
        await page.goto("http://localhost:8501/")
        # Wait for Streamlit to finish loading
        await page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30000)
        await page.wait_for_timeout(3000) # give it time to render components
        
        chat_path = os.path.join(OUTPUT_DIR, "1_chat_interface.png")
        await page.screenshot(path=chat_path, full_page=True)
        print(f"Saved {chat_path}")
        
        # 2. Charts Page
        print("Navigating to Charts Page...")
        await page.goto("http://localhost:8501/Charts")
        await page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30000)
        await page.wait_for_timeout(3000)
        
        charts_path = os.path.join(OUTPUT_DIR, "2_charts_page.png")
        await page.screenshot(path=charts_path, full_page=True)
        print(f"Saved {charts_path}")
        
        # 3. Reports Page
        print("Navigating to Reports Page...")
        await page.goto("http://localhost:8501/Reports")
        await page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30000)
        await page.wait_for_timeout(3000)
        
        reports_path = os.path.join(OUTPUT_DIR, "3_reports_page.png")
        await page.screenshot(path=reports_path, full_page=True)
        print(f"Saved {reports_path}")
        
        await browser.close()
        print("All screenshots captured successfully.")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
