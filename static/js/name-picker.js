/* BallCut - Name picker: a field for one name or for several, picked from the names used before.
   Typing filters the list; Enter takes the name the arrow keys point at, or adds what was typed when
   it is not in the list. A page has one list, shown under the field that has the focus.
   The review page uses it for the scorer and the assists of a score, for naming many scores at once,
   and for the list of names itself. */

let nameListElement = null;    // the list, shared by the pickers of the page
let nameListOwner = null;      // the picker whose names it shows

function cleanName(text) {
    return String(text == null ? '' : text).trim().replace(/\s+/g, ' ');
}

function createNamePicker(options) {
    const multiple = !!options.multiple;
    // For entering names only: once a name is taken the field is empty again. onChange gets that one
    // name; the names chosen are shown elsewhere, as buttons.
    const transient = !!options.transient;
    // Enter that took a name counts as "nothing more to type here": onEnter is called as well
    const advanceOnPick = !!options.advanceOnPick;
    const getNames = options.names || function () { return []; };
    const onChange = options.onChange || function () {};
    const onEnter = options.onEnter || null;
    const isExcluded = options.exclude || function () { return false; };

    let value = multiple ? [] : '';
    let typed = false;          // something was typed since the field got the focus or took a name
    let composing = false;      // an input method is choosing characters
    let choices = [];           // what the list shows: [{name, isNew}]
    let active = -1;            // the choice the arrow keys point at

    const el = document.createElement('div');
    el.className = 'name-field' + (multiple ? ' name-field-multi' : '');

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'name-input';
    input.placeholder = options.placeholder || '';
    input.maxLength = 30;
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-expanded', 'false');
    el.appendChild(input);

    function button(className, text, title) {
        const element = document.createElement('button');
        element.type = 'button';
        element.className = className;
        element.textContent = text;
        element.title = title;
        element.tabIndex = -1;
        element.addEventListener('mousedown', event => event.preventDefault());   // the field keeps the focus
        el.appendChild(element);
        return element;
    }

    let clear = null, toggle = null;
    if (!multiple) {
        clear = button('name-clear', '×', '清除');
        clear.addEventListener('click', () => removeValue(value));
        toggle = button('name-toggle', '▾', '显示名单');
        toggle.addEventListener('click', () => {
            if (nameListOwner === picker) return closeList();
            if (document.activeElement === input) openList(); else input.focus();
        });
    }

    function filterText() {
        return typed ? cleanName(input.value) : '';
    }

    function renderChoices() {
        const text = filterText();
        const all = getNames();
        choices = all.filter(name => !isExcluded(name) && (!text || name.toLowerCase().includes(text.toLowerCase())))
            .map(name => ({name, isNew: false}));
        // What was typed is not in the list (not spelled that way, at least): it can be added
        if (text && !all.some(name => name.toLowerCase() === text.toLowerCase())) {
            choices.push({name: text, isNew: true});
        }
        if (active >= choices.length) active = choices.length - 1;
    }

    function paintList() {
        const list = nameListElement;
        list.innerHTML = '';
        choices.forEach((choice, i) => {
            const row = document.createElement('div');
            row.className = 'name-option' + (choice.isNew ? ' name-option-new' : '') + (i === active ? ' is-active' : '');
            row.setAttribute('role', 'option');
            row.setAttribute('aria-selected', i === active ? 'true' : 'false');
            row.textContent = choice.isNew ? `＋ 新增「${choice.name}」` : choice.name;
            row.addEventListener('click', () => { take(choice); input.focus(); });
            list.appendChild(row);
        });
        if (active >= 0 && list.children[active]) {
            // The marked name into the visible part of the list (and nothing else: the list may be
            // partly outside the window before it is placed)
            const row = list.children[active];
            const top = row.offsetTop, bottom = top + row.offsetHeight;
            if (top < list.scrollTop) list.scrollTop = top;
            else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight;
        }
    }

    function placeList() {
        const rect = el.getBoundingClientRect();
        const list = nameListElement;
        list.style.minWidth = Math.max(200, Math.round(rect.width)) + 'px';
        list.style.left = Math.round(rect.left) + 'px';
        // Below the field; above it when there is more room there
        const below = window.innerHeight - rect.bottom;
        list.style.top = Math.round(below >= list.offsetHeight + 8 || below >= rect.top
                                    ? rect.bottom + 4 : rect.top - list.offsetHeight - 4) + 'px';
    }

    function openList() {
        renderChoices();
        if (!choices.length) return closeList();
        if (!nameListElement) nameListElement = makeListElement();
        if (nameListOwner && nameListOwner !== picker) nameListOwner.close();
        nameListOwner = picker;
        nameListElement.classList.remove('hidden');
        paintList();
        placeList();
        input.setAttribute('aria-expanded', 'true');
    }

    function closeList() {
        if (nameListOwner === picker) nameListOwner = null;
        active = -1;
        if (nameListElement) nameListElement.classList.add('hidden');
        input.setAttribute('aria-expanded', 'false');
    }

    function refreshList() {
        if (nameListOwner !== picker) return;
        renderChoices();
        if (!choices.length) return closeList();
        paintList();
        placeList();
    }

    // Takes a name into the field. Returns whether the names of the field changed by it.
    function add(rawName, picked) {
        const name = cleanName(rawName);
        if (!name || isExcluded(name)) return false;
        const had = multiple ? value.includes(name) : value === name;
        if (!transient && had) {
            typed = false;
            render();
            // Taking a name again with the mouse is still taking it
            if (picked) onChange(multiple ? value.slice() : value);
            return false;
        }
        value = multiple ? value.concat(name) : name;
        typed = false;
        if (transient) {
            input.value = '';
            onChange(name);
            value = multiple ? [] : '';
        } else {
            render();
            onChange(multiple ? value.slice() : value);
        }
        return true;
    }

    function take(choice) {
        add(choice.name, true);
        // A field for several goes on to show what is left to pick
        if (multiple && nameListOwner === picker) refreshList(); else closeList();
    }

    // What was typed becomes name(s). Returns whether the names of the field changed by it.
    function commitTyped() {
        const text = input.value;
        typed = false;
        input.value = '';
        let changed = false;
        for (const part of multiple ? text.split(/[,，、;；]+/) : [text]) {
            let name = cleanName(part);
            if (!name || isExcluded(name)) continue;
            // The same name spelled with other capital letters is the one that is in the list
            name = getNames().find(other => other.toLowerCase() === name.toLowerCase()) || name;
            changed = add(name) || changed;
        }
        // The text of a single name deleted down to nothing: the name is gone
        if (!changed && !multiple && !transient && !cleanName(text) && value) {
            removeValue('');
            changed = true;
        }
        return changed;
    }

    function removeValue(name) {
        value = multiple ? value.filter(other => other !== name) : '';
        render();
        onChange(multiple ? value.slice() : value);
    }

    function setValue(next) {
        value = multiple ? [...new Set((Array.isArray(next) ? next : []).map(cleanName).filter(Boolean))]
            : cleanName(next);
        typed = false;
        render();
    }

    function makeChip(name) {
        const chip = document.createElement('span');
        chip.className = 'name-chip';
        chip.textContent = name;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'name-chip-remove';
        remove.textContent = '×';
        remove.title = `移除 ${name}`;
        remove.setAttribute('aria-label', `移除 ${name}`);
        remove.addEventListener('mousedown', event => event.preventDefault());   // the field keeps the focus
        remove.addEventListener('click', () => removeValue(name));
        chip.appendChild(remove);
        return chip;
    }

    function render() {
        if (!multiple) {
            // Not while something is being typed there
            if (!(document.activeElement === input && typed)) input.value = value;
            el.classList.toggle('name-field-filled', !!value);
            return;
        }
        el.querySelectorAll('.name-chip').forEach(chip => chip.remove());
        if (!transient) for (const name of value) el.insertBefore(makeChip(name), input);
        if (document.activeElement !== input) input.value = '';
        input.placeholder = value.length ? '+' : (options.placeholder || '');
        el.classList.toggle('name-field-filled', value.length > 0);
    }

    input.addEventListener('compositionstart', () => {
        composing = true;
        closeList();
    });
    input.addEventListener('compositionend', () => {
        composing = false;
        typed = true;
        active = -1;
        openList();
    });
    input.addEventListener('input', () => {
        if (composing) return;
        typed = true;
        active = -1;
        openList();
    });
    input.addEventListener('focus', () => {
        typed = false;
        if (!composing) openList();
    });
    // Clicked into while it had the focus already (the list was closed with Enter or Escape)
    input.addEventListener('click', () => {
        if (nameListOwner !== picker && !composing) openList();
    });
    input.addEventListener('blur', () => {
        commitTyped();
        closeList();
    });
    input.addEventListener('keydown', event => {
        // An input method has the keyboard while it is choosing characters: its Enter confirms
        // characters, not a name
        if (event.isComposing || event.keyCode === 229) return;
        if (event.ctrlKey || event.metaKey || event.altKey) return;
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            if (nameListOwner !== picker) openList();
            if (!choices.length) return;
            active = (active + (event.key === 'ArrowDown' ? 1 : -1) + choices.length) % choices.length;
            paintList();
        } else if (event.key === 'Enter') {
            event.preventDefault();
            const taken = takeActive() || commitTyped();
            closeList();
            if (!taken || advanceOnPick) onEnter && onEnter();
        } else if (event.key === 'Tab') {
            commitTyped();     // what was typed becomes a name; the focus goes where the browser takes it
        } else if (event.key === 'Escape') {
            // The list first; the window around the field only when the list is closed
            if (nameListOwner === picker) {
                event.preventDefault();
                event.stopPropagation();
                closeList();
            }
        } else if (event.key === 'Backspace' && multiple && !transient && input.value === '' && value.length) {
            event.preventDefault();
            removeValue(value[value.length - 1]);
        } else if (multiple && /^[,，、;；]$/.test(event.key)) {
            event.preventDefault();
            commitTyped();
            if (nameListOwner === picker) refreshList();
        }
    });

    // The space of the field around its chips: a click there puts the cursor in it
    el.addEventListener('mousedown', event => {
        if (event.target === el) {
            event.preventDefault();
            input.focus();
        }
    });

    function takeActive() {
        const choice = nameListOwner === picker && active >= 0 ? choices[active] : null;
        if (!choice) return false;
        take(choice);
        return true;
    }

    const picker = {
        el,
        input,
        getValue: () => multiple ? value.slice() : value,
        setValue,
        focus() { input.focus(); },
        close: closeList,
        place: placeList,
        // For a field in a window that closes without the field losing the focus (closed with
        // Escape): what was typed there is not a name
        discard() {
            typed = false;
            input.value = '';
            closeList();
        },
        mark(row) {
            active = [...nameListElement.children].indexOf(row);
            paintList();
        }
    };
    render();
    return picker;
}

function makeListElement() {
    const list = document.createElement('div');
    list.className = 'name-list hidden';
    list.setAttribute('role', 'listbox');
    list.addEventListener('mousedown', event => event.preventDefault());   // the field keeps the focus
    list.addEventListener('mouseover', event => {
        const row = event.target.closest('.name-option');
        if (row && nameListOwner) nameListOwner.mark(row);
    });
    document.body.appendChild(list);
    // The page being scrolled or resized moves the field: the list follows and stays under it
    window.addEventListener('resize', () => nameListOwner && nameListOwner.place());
    window.addEventListener('scroll', event => {
        if (event.target !== list) nameListOwner && nameListOwner.place();
    }, true);
    return list;
}
