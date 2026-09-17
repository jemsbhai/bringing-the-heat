# Edit the slides

The editable PowerPoint is [`../output/Bringing-the-Heat.pptx`](../output/Bringing-the-Heat.pptx). It includes speaker notes, timings, and source links. You can edit it directly in PowerPoint or import it into Google Slides.

`slides.json` stores the slide text, notes, sources, and timing. `build-deck.mjs` creates the presentation from that content and the images in `../assets/` and `../showcase/assets/`. Generated previews and temporary build files belong outside the checked-in teaching materials.

The generator uses `@oai/artifact-tool`. It requires the Codex presentation runtime; it is not a standalone `npm install` project. Configure `SKILL_DIR` to the installed presentations skill, `RUNTIME_PYTHON` to its Python executable, and `RUNTIME_NODE_MODULES` to the bundled Node modules, then run `authoring/build-deck.mjs` with the bundled Node executable. See the script's opening comments for details. The exported PowerPoint and offline HTML work without the generator.

After changing content or layout, render every slide and check the result at presentation size. When changing the deck's cloud location, regenerate the QR image and verify that it decodes to the public viewing URL. The opening slide includes the talk title, Muntaser Syed, the website link, and the deck QR code.

The measured-results slide reports the saved laptop run. Colab outputs are separate measurements and should only replace it after updating the hardware, protocol, and validation notes alongside the numbers.
