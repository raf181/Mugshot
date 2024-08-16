> [!warning]
>As the developer of this tool, I provide it "as-is" without any warranties or guarantees of any kind. My responsibility extends only to the development  of the software. I do not assume any liability for how the tool is used, including any potential legal, ethical, or privacy issues arising from its usage.
> This project is licensed under the [GNU General Public License (GPL) v3.0](https://www.gnu.org/licenses/gpl-3.0.html). This license allows you to use, modify, and distribute the software under the terms specified in the GPL-3.0 license.

---

Now that we have gotten all the _"legal"_ talk out of  the way, these project was inspired by movies were intelligence services are traying to find data about someone and the have a magical database that gets images and data from a magical database that has all the necessary data. 

These is an idea for now is to make it _"modular"_ in how data is feed into it, so that custom data collectors can be build and runed separately form the server and having optional features

I'm thinking on having these process for now:
- Get pictures from a **_worker$^{[1]}$_** all the pictures in a local folder
- Retrieve the faces included in any picture and set change them to black and white
- Calculate the embeddings from the faces
- Store the embeddings in PostgreSQL in a `vector` column from `pgvector`
- Use `pgvector` distance function to retrieve the closest faces and therefore photos from people that look a like
- Having a validation process were Images are manually verified to get a more accurate profile for a specific individual of interest 



---
[1] - **Worker:** Module in charged of gathering and basic parsing images and sending them to the server.