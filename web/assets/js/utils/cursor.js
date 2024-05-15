import { app } from "../../../scripts/app.js";

/*
    Key functions from:
    https://github.com/yuku/textcomplete
    © Yuku Takahashi - This software is licensed under the MIT license.

    The MIT License (MIT)

    Copyright (c) 2015 Jonathan Ong me@jongleberry.com

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
const CHAR_CODE_ZERO = "0".charCodeAt(0);
const CHAR_CODE_NINE = "9".charCodeAt(0);

class NodeAreaHelper {
	constructor(el, getScale) {
		this.el = el;
		this.getScale = getScale;
	}

	#calculateElementOffset() {
		const rect = this.el.getBoundingClientRect();
		const owner = this.el.ownerDocument;
		if (owner == null) {
			throw new Error("Given element does not belong to document");
		}
		const { defaultView, documentElement } = owner;
		if (defaultView == null) {
			throw new Error("Given element does not belong to window");
		}
		const offset = {
			top: rect.top + defaultView.pageYOffset,
			left: rect.left + defaultView.pageXOffset,
		};
		if (documentElement) {
			offset.top -= documentElement.clientTop;
			offset.left -= documentElement.clientLeft;
		}
		return offset;
	}

	#isDigit(charCode) {
		return CHAR_CODE_ZERO <= charCode && charCode <= CHAR_CODE_NINE;
	}

	#getLineHeightPx() {
		const computedStyle = getComputedStyle(this.el);
		const lineHeight = computedStyle.lineHeight;
		// If the char code starts with a digit, it is either a value in pixels,
		// or unitless, as per:
		// https://drafts.csswg.org/css2/visudet.html#propdef-line-height
		// https://drafts.csswg.org/css2/cascade.html#computed-value
		if (this.#isDigit(lineHeight.charCodeAt(0))) {
			const floatLineHeight = parseFloat(lineHeight);
			// In real browsers the value is *always* in pixels, even for unit-less
			// line-heights. However, we still check as per the spec.
			return this.#isDigit(lineHeight.charCodeAt(lineHeight.length - 1))
				? floatLineHeight * parseFloat(computedStyle.fontSize)
				: floatLineHeight;
		}
		// Otherwise, the value is "normal".
		// If the line-height is "normal", calculate by font-size
		return this.#calculateLineHeightPx(this.el.nodeName, computedStyle);
	}

	/**
	 * Returns calculated line-height of the given node in pixels.
	 */
	#calculateLineHeightPx(nodeName, computedStyle) {
		const body = document.body;
		if (!body) return 0;

		const tempNode = document.createElement(nodeName);
		tempNode.innerHTML = "&nbsp;";
		Object.assign(tempNode.style, {
			fontSize: computedStyle.fontSize,
			fontFamily: computedStyle.fontFamily,
			padding: "0",
			position: "absolute",
		});
		body.appendChild(tempNode);

		// Make sure textarea has only 1 row
		if (tempNode instanceof HTMLTextAreaElement) {
			tempNode.rows = 1;
		}

		// Assume the height of the element is the line-height
		const height = tempNode.offsetHeight;
		body.removeChild(tempNode);

		return height;
	}

	getCursorOffset() {
		const scale = this.getScale();
		const elOffset = this.#calculateElementOffset();
		const elScroll = this.#getElScroll();
		const cursorPosition = this.#getCursorPosition();
		const lineHeight = this.#getLineHeightPx();
		const top = elOffset.top - (elScroll.top * scale) + (cursorPosition.top + lineHeight) * scale;
		const left = elOffset.left - elScroll.left + cursorPosition.left;
		const clientTop = this.el.getBoundingClientRect().top;
		if (this.el.dir !== "rtl") {
			return { top, left, lineHeight, clientTop };
		} else {
			const right = document.documentElement ? document.documentElement.clientWidth - left : 0;
			return { top, right, lineHeight, clientTop };
		}
	}

	#getElScroll() {
		return { top: this.el.scrollTop, left: this.el.scrollLeft };
	}

	#getCursorPosition() {
		return getCaretCoordinates(this.el, this.el.selectionEnd);
	}

	getBeforeCursor() {
		return this.el.selectionStart !== this.el.selectionEnd ? null : this.el.value.substring(0, this.el.selectionEnd);
	}

	getAfterCursor() {
		return this.el.value.substring(this.el.selectionEnd);
	}

	insertAtCursor(value, offset, finalOffset) {
		if (this.el.selectionStart != null) {
			const startPos = this.el.selectionStart;
			const endPos = this.el.selectionEnd;

			// Move selection to beginning of offset
			this.el.selectionStart = this.el.selectionStart + offset;

			// Using execCommand to support undo, but since it's officially 
			// 'deprecated' we need a backup solution, but it won't support undo :(
			let pasted = true;
			try {
				if (!document.execCommand("insertText", false, value)) {
					pasted = false;
				}
			} catch (e) {
				console.error("Error caught during execCommand:", e);
				pasted = false;
			}

			if (!pasted) {
				console.error(
					"execCommand unsuccessful; not supported. Adding text manually, no undo support.");
				textarea.setRangeText(modifiedText, this.el.selectionStart, this.el.selectionEnd, 'end');
			}

			this.el.selectionEnd = this.el.selectionStart = startPos + value.length + offset + (finalOffset ?? 0);
		} else {
			// Using execCommand to support undo, but since it's officially 
			// 'deprecated' we need a backup solution, but it won't support undo :(
			let pasted = true;
			try {
				if (!document.execCommand("insertText", false, value)) {
					pasted = false;
				}
			} catch (e) {
				console.error("Error caught during execCommand:", e);
				pasted = false;
			}

			if (!pasted) {
				console.error(
					"execCommand unsuccessful; not supported. Adding text manually, no undo support.");
				this.el.value += value;
			}
		}
	}
}

/*********************/

/**
 * @typedef {{
* 	text: string,
* 	priority?: number,
* 	info?: Function,
* 	hint?: string,
*  showValue?: boolean,
*  caretOffset?: number
* }} AutoCompleteEntry
*/
export class cursor {

	/**
	 * @param {HTMLTextAreaElement} el
	 */
	constructor(el) {
		this.el = el;
		this.helper = new NodeAreaHelper(el, () => app.canvas.ds.scale);

		this.#setup();
	}

	#setup() {
		this.el.addEventListener("click", this.#hide.bind(this));
	}

    #hide() {
	}

}
