// mgeHost64 — the knob registry. See knobs.h for why this exists.
//
// It is deliberately a dumb container with no policy in it. All the JUDGEMENT about knobs —
// the MGE_HOST_KNOBS parse, the clamp message, the "handled in main.cpp" exception list, the
// pointer -> original-value map the dev panel resets from — stays in forgerender.cpp, because
// every one of those is about how this project measures things and none of it is about storage.
// Keeping the split there is what lets this file be read in one sitting and never need changing
// when a knob is added.

#include "knobs.h"

#include <cstring>
#include <vector>

namespace Knobs {

    namespace {

        // ⚠ A FUNCTION-LOCAL STATIC, NOT A FILE-SCOPE CONTAINER, AND THAT IS THE WHOLE TRICK.
        // Knobs register from static initialisers in other translation units, and the order those
        // run in across TUs is unspecified — so a file-scope container here could still be
        // uninitialised when the first add() lands, and that registration would be written into a
        // vector that is about to be constructed over the top of it. A function-local static is
        // constructed on first use by definition, which is the first add() whoever it comes from.
        // This is the static-initialisation-order fiasco and it is the one real hazard in this file.
        //
        // ⚠ IT IS NEVER DESTROYED, also deliberately. The same ordering problem runs in reverse at
        // exit: a TU's teardown could destroy the table while another TU's destructor is still
        // reading it. The table is small, the process is ending, and leaking it is a strictly
        // better trade than a destruction-order crash in a render host's shutdown path.
        //
        // ⚠ ENTRIES ARE HEAP-ALLOCATED AND THE TABLE HOLDS POINTERS, so find() and nameOf() can
        // hand out an `Entry*` that stays valid forever. A vector<Entry> would reallocate on the
        // next registration and silently dangle every pointer it had already returned. Today every
        // registration happens before main() and every lookup after, so that would work by
        // accident — but "works as long as nobody registers a knob late" is exactly the kind of
        // unwritten precondition this refactor exists to stop relying on.
        std::vector<Entry*>& table() {
            static std::vector<Entry*>* t = [] {
                auto* v = new std::vector<Entry*>();
                v->reserve(320);          // 277 today; one allocation instead of nine
                return v;
            }();
            return *t;
        }

        int g_dupNames = 0;

        bool push(const Entry& e) {
            std::vector<Entry*>& t = table();
            for (const Entry* x : t) {
                if (std::strcmp(x->name, e.name) == 0) {
                    // Counted, not rejected. There is nowhere to report at static-init time — no
                    // log is open yet, and throwing from a static initialiser takes the process
                    // down before anything can say why. forgerender.cpp reports the count once the
                    // log exists, and --knob-dump names the offender.
                    ++g_dupNames;
                    break;
                }
            }
            t.push_back(new Entry(e));
            return true;
        }

    }  // namespace

    bool add(const char* name, float* p)    { return push({ name, p, 0u, 0u, KindF }); }
    bool add(const char* name, bool* p)     { return push({ name, p, 0u, 0u, KindB }); }
    bool add(const char* name, uint32_t* p, uint32_t umax) {
        return push({ name, p, umax, 0u, KindU });
    }
    bool add(const char* name, char* p, size_t cap) {
        return push({ name, p, 0u, (uint32_t)cap, KindS });
    }

    const Entry* find(const char* name) {
        if (!name) { return nullptr; }
        for (const Entry* e : table()) {
            if (std::strcmp(e->name, name) == 0) { return e; }
        }
        return nullptr;
    }

    const char* nameOf(const void* p) {
        if (!p) { return nullptr; }
        for (const Entry* e : table()) {
            if (e->p == p) { return e->name; }
        }
        return nullptr;
    }

    size_t       count()      { return table().size(); }
    const Entry* at(size_t i) { return (i < table().size()) ? table()[i] : nullptr; }
    int          dupNames()   { return g_dupNames; }

}  // namespace Knobs
