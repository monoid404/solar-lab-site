# SOLAR LAB | GREEN NARAE

React homepage for the solar panel output prediction research dashboard.

## Run locally

1. Install Node.js.
2. Open this folder in a terminal and run `npm install`.
3. Run `npm run dev` and open the local address shown in the terminal.

Tailwind CSS is loaded from its CDN. The prediction feature reads a CSV in the browser and trains the model locally. The CSV header must include `온도`, `습도`, `밝기`, and `태양광 패널 출력 전압`.

## Publish a public link with GitHub Pages

1. Create a new **public** repository on GitHub, for example `solar-lab-site`. Do not add a README or other starter files.
2. Install GitHub Desktop, choose **File → Add Local Repository**, and select this `solar-lab-site` folder. If GitHub Desktop says it is not a repository yet, choose **create a repository here**.
3. In GitHub Desktop, commit the files and choose **Publish repository**. Keep the repository public.
4. On GitHub, open the repository's **Settings → Pages** and choose **GitHub Actions** under **Build and deployment → Source**.
5. The included workflow builds and publishes the site whenever you push to `main`. Check the **Actions** tab; when the deployment finishes, open **Settings → Pages** to get the public URL.

The published page is public to anyone with the link. Later edits go live after you commit and push them from GitHub Desktop.
